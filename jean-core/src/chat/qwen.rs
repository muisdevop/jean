//! Qwen Code headless execution engine.
//!
//! Runs `qwen -p <prompt> --output-format stream-json` and maps its
//! Claude-style JSONL events (`system` / `assistant` / `user` / `result`)
//! onto Jean's chat events. Qwen Code is a single attached process, so a
//! cancelled turn is killed through the shared process registry.

use super::coalesce::ChunkCoalescer;
use super::types::{ContentBlock, ToolCall, UsageData};
use crate::http_server::EmitExt;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Stdio;
use tauri::AppHandle;

pub struct QwenResponse {
    pub content: String,
    pub session_id: String,
    pub tool_calls: Vec<ToolCall>,
    pub content_blocks: Vec<ContentBlock>,
    pub cancelled: bool,
    pub usage: Option<UsageData>,
}

pub struct QwenExecutionOptions<'a> {
    pub app: &'a AppHandle,
    pub jean_session_id: &'a str,
    pub worktree_id: &'a str,
    pub working_dir: &'a Path,
    pub existing_qwen_session_id: Option<&'a str>,
    pub model: Option<&'a str>,
    pub execution_mode: Option<&'a str>,
    pub message: &'a str,
    pub system_prompt: Option<&'a str>,
    pub pid_callback: Option<Box<dyn FnOnce(u32) + Send>>,
}

fn emit(app: &AppHandle, event: &str, value: Value) {
    let _ = app.emit_all(event, &value);
}

fn emit_chunk(app: &AppHandle, session_id: &str, worktree_id: &str, content: &str) {
    emit(
        app,
        "chat:chunk",
        json!({
            "session_id": session_id,
            "worktree_id": worktree_id,
            "content": content,
        }),
    );
}

fn flush_chunks(
    app: &AppHandle,
    session_id: &str,
    worktree_id: &str,
    coalescer: &mut ChunkCoalescer,
) {
    if let Some(batch) = coalescer.flush() {
        emit_chunk(app, session_id, worktree_id, &batch);
    }
}

/// Qwen Code takes a single `-p` prompt, so Jean's system instructions are
/// wrapped into the message rather than passed as a separate flag.
fn prepared_message(message: &str, system_prompt: Option<&str>) -> String {
    match system_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(prompt) => {
            format!("<system_instructions>\n{prompt}\n</system_instructions>\n\n{message}")
        }
        None => message.to_string(),
    }
}

/// Extract plain text from a string or an array of content blocks.
fn text_from_content(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("thinking").and_then(Value::as_str))
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn usage_from(value: &Value) -> Option<UsageData> {
    let read = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| value.get(*name).and_then(Value::as_u64))
            .unwrap_or(0)
    };
    let usage = UsageData {
        input_tokens: read(&["input_tokens", "inputTokens"]),
        output_tokens: read(&["output_tokens", "outputTokens"]),
        cache_read_input_tokens: read(&["cache_read_input_tokens", "cachedTokens"]),
        cache_creation_input_tokens: read(&["cache_creation_input_tokens"]),
    };
    if usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cache_read_input_tokens == 0
        && usage.cache_creation_input_tokens == 0
    {
        None
    } else {
        Some(usage)
    }
}

fn append_text(response: &mut QwenResponse, text: &str) {
    if text.is_empty() {
        return;
    }
    response.content.push_str(text);
    match response.content_blocks.last_mut() {
        Some(ContentBlock::Text { text: existing }) => existing.push_str(text),
        _ => response
            .content_blocks
            .push(ContentBlock::Text { text: text.to_string() }),
    }
}

fn handle_assistant(app: &AppHandle, session_id: &str, worktree_id: &str, value: &Value, response: &mut QwenResponse, saw_partial_text: bool) {
    let Some(content) = value.pointer("/message/content") else {
        return;
    };
    let Some(blocks) = content.as_array() else {
        return;
    };
    for block in blocks {
        match block.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => {
                if saw_partial_text {
                    continue;
                }
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    append_text(response, text);
                    emit_chunk(app, session_id, worktree_id, text);
                }
            }
            "thinking" => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    emit(
                        app,
                        "chat:thinking",
                        json!({
                            "session_id": session_id,
                            "worktree_id": worktree_id,
                            "content": text,
                        }),
                    );
                }
            }
            "tool_use" => {
                let Some(id) = block.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("Tool")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                response.tool_calls.push(ToolCall {
                    id: id.to_string(),
                    name: name.clone(),
                    input: input.clone(),
                    output: None,
                    parent_tool_use_id: None,
                    is_error: None,
                    subagent_usage: None,
                });
                response
                    .content_blocks
                    .push(ContentBlock::ToolUse {
                        tool_call_id: id.to_string(),
                    });
                emit(
                    app,
                    "chat:tool_use",
                    json!({
                        "session_id": session_id,
                        "worktree_id": worktree_id,
                        "id": id,
                        "name": name,
                        "input": input,
                    }),
                );
            }
            _ => {}
        }
    }
}

fn handle_tool_results(app: &AppHandle, session_id: &str, worktree_id: &str, value: &Value, response: &mut QwenResponse) {
    let Some(blocks) = value.pointer("/message/content").and_then(Value::as_array) else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        let Some(id) = block
            .get("tool_use_id")
            .or_else(|| block.get("toolUseId"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let output = block
            .get("content")
            .map(text_from_content)
            .unwrap_or_default();
        let is_error = block.get("is_error").and_then(Value::as_bool);
        if let Some(call) = response.tool_calls.iter_mut().find(|call| call.id == id) {
            call.output = Some(output.clone());
            call.is_error = is_error;
        }
        emit(
            app,
            "chat:tool_result",
            json!({
                "session_id": session_id,
                "worktree_id": worktree_id,
                "tool_use_id": id,
                "output": output,
            }),
        );
    }
}

pub fn execute_qwen(options: QwenExecutionOptions<'_>) -> Result<QwenResponse, String> {
    let binary = crate::qwen_cli::resolve_cli_binary(options.app);
    if !crate::qwen_cli::binary_exists(&binary) {
        return Err(
            "Qwen Code CLI not installed. Install it from Settings or run `npm install -g @qwen-code/qwen-code`."
                .to_string(),
        );
    }

    let mut command =
        crate::platform::cli_command(&binary.to_string_lossy(), Some(options.working_dir));
    command
        .arg("-p")
        .arg(prepared_message(options.message, options.system_prompt))
        .arg("--output-format")
        .arg("stream-json")
        .arg("--include-partial-messages");
    if let Some(model) = options.model.map(str::trim).filter(|model| !model.is_empty()) {
        command.arg("--model").arg(model);
    }
    if let Some(resume_id) = options
        .existing_qwen_session_id
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        command.arg("--resume").arg(resume_id);
    }
    match options.execution_mode.unwrap_or("build") {
        "yolo" => {
            command.arg("--yolo");
        }
        "plan" => {
            command.arg("--approval-mode").arg("plan");
        }
        _ => {
            command.arg("--approval-mode").arg("auto-edit");
        }
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to start Qwen Code: {error}"))?;
    let pid = child.id();
    if let Some(callback) = options.pid_callback {
        callback(pid);
    }
    if !super::registry::register_process(options.jean_session_id.to_string(), pid) {
        let _ = crate::platform::kill_process_tree(pid);
        let _ = crate::platform::kill_process(pid);
        return Ok(QwenResponse {
            content: String::new(),
            session_id: String::new(),
            tool_calls: Vec::new(),
            content_blocks: Vec::new(),
            cancelled: true,
            usage: None,
        });
    }

    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                log::debug!("[qwen] {line}");
            }
        });
    }

    let mut response = QwenResponse {
        content: String::new(),
        session_id: String::new(),
        tool_calls: Vec::new(),
        content_blocks: Vec::new(),
        cancelled: false,
        usage: None,
    };
    let mut coalescer = ChunkCoalescer::new();
    let mut saw_partial_text = false;
    let mut saw_result = false;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture Qwen Code stdout".to_string())?;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            log::debug!("[qwen] non-JSON line: {line}");
            continue;
        };
        match value.get("type").and_then(Value::as_str).unwrap_or_default() {
            "system" => {
                if let Some(session_id) = value.get("session_id").and_then(Value::as_str) {
                    response.session_id = session_id.to_string();
                }
            }
            "assistant" => handle_assistant(
                options.app,
                options.jean_session_id,
                options.worktree_id,
                &value,
                &mut response,
                saw_partial_text,
            ),
            "user" => handle_tool_results(
                options.app,
                options.jean_session_id,
                options.worktree_id,
                &value,
                &mut response,
            ),
            "stream_event" => {
                let event = value.get("event").unwrap_or(&Value::Null);
                let is_text_delta = event.get("type").and_then(Value::as_str)
                    == Some("content_block_delta")
                    && event.pointer("/delta/type").and_then(Value::as_str) == Some("text_delta");
                if is_text_delta {
                    if let Some(text) = event.pointer("/delta/text").and_then(Value::as_str) {
                        saw_partial_text = true;
                        append_text(&mut response, text);
                        if let Some(batch) = coalescer.push(text) {
                            emit_chunk(
                                options.app,
                                options.jean_session_id,
                                options.worktree_id,
                                &batch,
                            );
                        }
                    }
                }
            }
            "result" => {
                saw_result = true;
                flush_chunks(
                    options.app,
                    options.jean_session_id,
                    options.worktree_id,
                    &mut coalescer,
                );
                if response.session_id.is_empty() {
                    if let Some(session_id) = value.get("session_id").and_then(Value::as_str) {
                        response.session_id = session_id.to_string();
                    }
                }
                if let Some(result) = value.get("result").and_then(Value::as_str) {
                    if response.content.trim().is_empty() && !result.trim().is_empty() {
                        response.content = result.to_string();
                        response
                            .content_blocks
                            .push(ContentBlock::Text {
                                text: result.to_string(),
                            });
                        emit_chunk(
                            options.app,
                            options.jean_session_id,
                            options.worktree_id,
                            result,
                        );
                    }
                }
                if let Some(usage) = value.get("usage").and_then(usage_from) {
                    response.usage = Some(usage);
                }
                if value.get("is_error").and_then(Value::as_bool) == Some(true) {
                    emit(
                        options.app,
                        "chat:error",
                        json!({
                            "session_id": options.jean_session_id,
                            "worktree_id": options.worktree_id,
                            "error": value
                                .get("result")
                                .and_then(Value::as_str)
                                .unwrap_or("Qwen Code returned an error"),
                        }),
                    );
                }
            }
            _ => {}
        }
    }

    flush_chunks(
        options.app,
        options.jean_session_id,
        options.worktree_id,
        &mut coalescer,
    );

    let cancelled =
        !saw_result && !super::registry::is_process_registered(options.jean_session_id, pid);
    let _ = child.wait();
    super::registry::unregister_process(options.jean_session_id);

    response.content = response.content.trim().to_string();
    response.cancelled = cancelled;
    emit(
        options.app,
        "chat:done",
        json!({
            "session_id": options.jean_session_id,
            "worktree_id": options.worktree_id,
            "waiting_for_plan": false,
        }),
    );
    Ok(response)
}

/// One-shot execution for magic prompts. Returns the final assistant text.
pub fn execute_one_shot_qwen(
    app: &AppHandle,
    working_dir: &Path,
    message: &str,
    model: Option<&str>,
) -> Result<String, String> {
    let binary = crate::qwen_cli::resolve_cli_binary(app);
    if !crate::qwen_cli::binary_exists(&binary) {
        return Err("Qwen Code CLI not installed".to_string());
    }
    let mut command = crate::platform::cli_command(&binary.to_string_lossy(), Some(working_dir));
    command
        .arg("-p")
        .arg(message)
        .arg("--output-format")
        .arg("json");
    if let Some(model) = model.map(str::trim).filter(|model| !model.is_empty()) {
        command.arg("--model").arg(model);
    }
    let output = command
        .output()
        .map_err(|error| format!("Failed to run Qwen Code: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Qwen Code failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let value: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Failed to parse Qwen Code output: {error}"))?;
    let result = value
        .as_array()
        .and_then(|items| items.iter().rev().find_map(|item| {
            item.get("result").and_then(Value::as_str).map(str::to_string)
        }))
        .or_else(|| {
            value
                .get("result")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| "Qwen Code returned no result".to_string())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_system_prompt_into_the_prompt() {
        let prepared = prepared_message("hello", Some("be terse"));
        assert!(prepared.contains("<system_instructions>\nbe terse\n</system_instructions>"));
        assert!(prepared.ends_with("hello"));
        assert_eq!(prepared_message("hello", None), "hello");
    }

    #[test]
    fn extracts_text_from_strings_and_blocks() {
        assert_eq!(text_from_content(&json!("hi")), "hi");
        assert_eq!(
            text_from_content(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])),
            "ab"
        );
    }

    #[test]
    fn reads_snake_and_camel_case_usage() {
        let usage = usage_from(&json!({"input_tokens": 3, "outputTokens": 4})).unwrap();
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 4);
        assert!(usage_from(&json!({})).is_none());
    }
}
