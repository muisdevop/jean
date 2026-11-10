use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tauri::AppHandle;

use super::config::{
    binary_exists, ensure_cli_dir, find_system_qwen_binary, get_cli_binary_path, get_cli_dir,
    resolve_cli_binary,
};

const PACKAGE_NAME: &str = "@qwen-code/qwen-code";
const NPM_REGISTRY_URL: &str = "https://registry.npmjs.org/%40qwen-code%2Fqwen-code";

/// Models Qwen Code ships as defaults when its settings file has no model list.
const DEFAULT_MODELS: &[(&str, &str)] = &[
    ("qwen3-coder-plus", "Qwen3 Coder Plus"),
    ("qwen3-coder-flash", "Qwen3 Coder Flash"),
    ("qwen3.8-max", "Qwen 3.8 Max"),
    ("qwen3.7-plus", "Qwen 3.7 Plus"),
    ("qwen3.8-flash", "Qwen 3.8 Flash"),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenCliStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenAuthStatus {
    pub authenticated: bool,
    pub error: Option<String>,
    #[serde(default)]
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenPathDetection {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
    pub package_manager: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenModelInfo {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenInstallCommand {
    pub command: String,
    pub args: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenReleaseInfo {
    pub version: String,
    pub tag_name: String,
    pub published_at: String,
    pub prerelease: bool,
}

fn package(version: Option<&str>) -> String {
    match version.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("latest") => format!("{PACKAGE_NAME}@latest"),
        Some(value) if value.starts_with("@qwen-code/qwen-code@") => value.to_string(),
        Some(value) => format!("{PACKAGE_NAME}@{value}"),
    }
}

fn parse_version(bytes: &[u8]) -> Option<String> {
    String::from_utf8_lossy(bytes)
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
        .map(|part| part.trim_start_matches('v').to_string())
}

fn semver_parts(version: &str) -> Vec<u32> {
    version
        .split(['-', '+'])
        .next()
        .unwrap_or(version)
        .split('.')
        .map(|part| part.parse().unwrap_or_default())
        .collect()
}

fn qwen_config_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|home| home.join(".qwen"))
}

fn read_qwen_settings() -> Option<Value> {
    let path = qwen_config_dir()?.join("settings.json");
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

/// Qwen Code has no non-interactive auth-status command, so Jean inspects the
/// credentials Qwen Code itself writes: `~/.qwen/oauth_creds.json` for Qwen
/// OAuth, a configured API key in `settings.json`, or a provider key in the
/// environment. No network call and no token spend.
fn check_auth() -> QwenAuthStatus {
    if let Some(dir) = qwen_config_dir() {
        let oauth = dir.join("oauth_creds.json");
        if let Ok(contents) = std::fs::read_to_string(&oauth) {
            if let Ok(value) = serde_json::from_str::<Value>(&contents) {
                let has_token = value
                    .get("access_token")
                    .or_else(|| value.get("accessToken"))
                    .and_then(Value::as_str)
                    .is_some_and(|token| !token.trim().is_empty());
                if has_token {
                    return QwenAuthStatus {
                        authenticated: true,
                        error: None,
                        timed_out: false,
                    };
                }
            }
        }
    }

    if let Some(settings) = read_qwen_settings() {
        if settings_has_api_key(&settings) {
            return QwenAuthStatus {
                authenticated: true,
                error: None,
                timed_out: false,
            };
        }
    }

    for key in [
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ] {
        if std::env::var(key).is_ok_and(|value| !value.trim().is_empty()) {
            return QwenAuthStatus {
                authenticated: true,
                error: None,
                timed_out: false,
            };
        }
    }

    QwenAuthStatus {
        authenticated: false,
        error: Some(
            "Not signed in. Run `qwen` and use /auth, or set QWEN_API_KEY / OPENAI_API_KEY."
                .to_string(),
        ),
        timed_out: false,
    }
}

fn settings_has_api_key(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.iter().any(|(key, child)| {
            (key.eq_ignore_ascii_case("apiKey")
                || key.eq_ignore_ascii_case("api_key")
                || key.eq_ignore_ascii_case("authToken"))
                && child.as_str().is_some_and(|text| !text.trim().is_empty())
                || settings_has_api_key(child),
        }),
        Value::Array(items) => items.iter().any(settings_has_api_key),
        _ => false,
    }
}

fn parse_models(settings: Option<&Value>) -> Vec<QwenModelInfo> {
    let default_model = settings
        .and_then(|value| value.pointer("/model/name"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut models: Vec<QwenModelInfo> = Vec::new();

    if let Some(providers) = settings
        .and_then(|value| value.get("modelProviders"))
        .and_then(Value::as_object)
    {
        for provider in providers.values() {
            let Some(list) = provider.get("models").and_then(Value::as_array) else {
                continue;
            };
            for model in list {
                let Some(id) = model.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let label = model
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(id)
                    .to_string();
                models.push(QwenModelInfo {
                    id: id.to_string(),
                    label,
                    is_default: default_model.as_deref() == Some(id),
                });
            }
        }
    }

    if models.is_empty() {
        models = DEFAULT_MODELS
            .iter()
            .map(|(id, label)| QwenModelInfo {
                id: (*id).to_string(),
                label: (*label).to_string(),
                is_default: default_model.as_deref() == Some(*id),
            })
            .collect();
    }

    models.sort_by(|left, right| {
        right
            .is_default
            .cmp(&left.is_default)
            .then_with(|| left.label.cmp(&right.label))
    });
    models.dedup_by(|left, right| left.id == right.id);
    models
}

pub async fn check_qwen_cli_installed(app: AppHandle) -> Result<QwenCliStatus, String> {
    let binary = resolve_cli_binary(&app);
    if !binary_exists(&binary) {
        return Ok(QwenCliStatus {
            installed: false,
            version: None,
            path: None,
        });
    }
    let version = crate::platform::cli_command(&binary.to_string_lossy(), None)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| parse_version(&output.stdout));
    Ok(QwenCliStatus {
        installed: true,
        version,
        path: Some(binary.to_string_lossy().to_string()),
    })
}

pub async fn detect_qwen_in_path(app: AppHandle) -> Result<QwenPathDetection, String> {
    let Some(binary) = find_system_qwen_binary(&app) else {
        return Ok(QwenPathDetection {
            found: false,
            path: None,
            version: None,
            package_manager: None,
        });
    };
    let version = crate::platform::cli_command(&binary.to_string_lossy(), None)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| parse_version(&output.stdout));
    Ok(QwenPathDetection {
        found: true,
        path: Some(binary.to_string_lossy().to_string()),
        version,
        package_manager: Some("path".to_string()),
    })
}

pub async fn check_qwen_cli_auth(app: AppHandle) -> Result<QwenAuthStatus, String> {
    let binary = resolve_cli_binary(&app);
    if !binary_exists(&binary) {
        return Ok(QwenAuthStatus {
            authenticated: false,
            error: Some("Qwen Code CLI not installed".to_string()),
            timed_out: false,
        });
    }
    Ok(check_auth())
}

pub async fn list_qwen_models(app: AppHandle) -> Result<Vec<QwenModelInfo>, String> {
    let binary = resolve_cli_binary(&app);
    if !binary_exists(&binary) {
        return Ok(Vec::new());
    }
    Ok(parse_models(read_qwen_settings().as_ref()))
}

pub async fn get_available_qwen_versions(_app: AppHandle) -> Result<Vec<QwenReleaseInfo>, String> {
    let value: Value = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|error| format!("Failed to build Qwen HTTP client: {error}"))?
        .get(NPM_REGISTRY_URL)
        .send()
        .await
        .map_err(|error| format!("Failed to fetch Qwen Code versions: {error}"))?
        .json()
        .await
        .map_err(|error| format!("Failed to parse Qwen Code versions: {error}"))?;
    let latest = value
        .pointer("/dist-tags/latest")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut versions = value
        .get("versions")
        .and_then(Value::as_object)
        .map(|versions| {
            versions
                .keys()
                .map(|version| QwenReleaseInfo {
                    version: version.clone(),
                    tag_name: if version == latest { "latest" } else { version }.to_string(),
                    published_at: String::new(),
                    prerelease: version.contains('-'),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    versions.sort_by_key(|version| std::cmp::Reverse(semver_parts(&version.version)));
    Ok(versions)
}

pub async fn check_qwen_cli_version_exists(
    _app: AppHandle,
    version: String,
) -> Result<bool, String> {
    let version = version.trim().trim_start_matches('v');
    if version.is_empty() {
        return Ok(false);
    }
    let value: Value = reqwest::get(NPM_REGISTRY_URL)
        .await
        .map_err(|error| format!("Failed to fetch Qwen Code versions: {error}"))?
        .json()
        .await
        .map_err(|error| format!("Failed to parse Qwen Code versions: {error}"))?;
    Ok(value
        .get("versions")
        .and_then(Value::as_object)
        .is_some_and(|versions| versions.contains_key(version)))
}

pub async fn get_qwen_install_command(app: AppHandle) -> Result<QwenInstallCommand, String> {
    let dir = get_cli_dir(&app)?;
    Ok(QwenInstallCommand {
        command: "npm".to_string(),
        args: vec![
            "install".to_string(),
            "--prefix".to_string(),
            dir.to_string_lossy().to_string(),
            package(None),
        ],
        description: "Install Qwen Code into Jean's managed app-data directory".to_string(),
    })
}

pub async fn install_qwen_cli(app: AppHandle, version: Option<String>) -> Result<(), String> {
    let npm_path = crate::prerequisites::require_npm("Qwen Code CLI")?;
    let dir = ensure_cli_dir(&app)?;
    let output = crate::platform::host_cli_command(&npm_path, None)
        .args(["install", "--prefix"])
        .arg(&dir)
        .arg(package(version.as_deref()))
        .output()
        .map_err(|error| format!("Failed to install Qwen Code: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Qwen Code install failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    if !get_cli_binary_path(&app)?.exists() {
        return Err("Qwen Code install completed but the binary was not found".to_string());
    }
    crate::expose_managed_cli("qwen", &get_cli_binary_path(&app)?);
    Ok(())
}

pub async fn uninstall_qwen_cli(app: AppHandle) -> Result<(), String> {
    let dir = get_cli_dir(&app)?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .map_err(|error| format!("Failed to remove Qwen Code CLI: {error}"))?;
    }
    Ok(())
}

pub async fn update_qwen_cli(app: AppHandle) -> Result<(), String> {
    install_qwen_cli(app, None).await
}

/// Qwen Code has no non-interactive login command: `/auth` runs inside the
/// interactive TUI. Jean opens a terminal running `qwen` for that flow instead,
/// so this command only reports how to sign in.
pub async fn login_qwen_cli(_app: AppHandle) -> Result<(), String> {
    Err(
        "Qwen Code signs in interactively. Open a terminal, run `qwen`, and use /auth, or set QWEN_API_KEY."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_uses_official_qwen_code_package() {
        assert_eq!(package(None), "@qwen-code/qwen-code@latest");
        assert_eq!(package(Some("0.27.1")), "@qwen-code/qwen-code@0.27.1");
    }

    #[test]
    fn parses_cli_version() {
        assert_eq!(
            parse_version(b"0.27.1\n").as_deref(),
            Some("0.27.1")
        );
    }

    #[test]
    fn detects_api_keys_in_nested_settings() {
        let settings = serde_json::json!({
            "modelProviders": {
                "openai": {"apiKey": "sk-test", "models": [{"id": "qwen3-coder-plus"}]}
            }
        });
        assert!(settings_has_api_key(&settings));
        assert!(!settings_has_api_key(&serde_json::json!({"model": {"name": "x"}})));
    }

    #[test]
    fn falls_back_to_default_models_without_settings() {
        let models = parse_models(None);
        assert!(models.iter().any(|model| model.id == "qwen3-coder-plus"));
        assert!(models.iter().any(|model| model.id == "qwen3.8-max"));
    }

    #[test]
    fn prefers_configured_models_and_marks_the_default() {
        let settings = serde_json::json!({
            "model": {"name": "custom-coder"},
            "modelProviders": {
                "qwen": {"models": [
                    {"id": "other", "name": "Other"},
                    {"id": "custom-coder", "name": "Custom Coder"}
                ]}
            }
        });
        let models = parse_models(Some(&settings));
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "custom-coder");
        assert!(models[0].is_default);
        assert_eq!(models[0].label, "Custom Coder");
    }
}
