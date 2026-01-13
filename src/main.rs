use teloxide::prelude::*;
use chrono::{Local, Duration, NaiveTime};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Deserialize)]
struct YouTubeSearchResponse {
    items: Vec<YouTubeVideo>,
}

#[derive(Debug, Deserialize)]
struct YouTubeVideo {
    id: VideoId,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VideoId {
    Video { #[serde(rename = "videoId")] video_id: String },
    String(String),
}

// Groq API структуры
#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

// Структура для хранения истории сообщений
#[derive(Debug, Clone)]
struct ChatMessage {
    author: String,
    text: String,
}

struct ChatHistory {
    messages: VecDeque<ChatMessage>,
    max_size: usize,
}

impl ChatHistory {
    fn new(max_size: usize) -> Self {
        Self {
            messages: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add_message(&mut self, author: String, text: String) {
        if self.messages.len() >= self.max_size {
            self.messages.pop_front();
        }
        self.messages.push_back(ChatMessage { author, text });

        // Логируем в формате "Автор - Сообщение"
        log::info!("HISTORY: {} - {}",
            self.messages.back().unwrap().author,
            self.messages.back().unwrap().text
        );
    }

    fn format_for_prompt(&self) -> String {
        if self.messages.is_empty() {
            return String::new();
        }

        let mut history = String::from("\n\n<История переписки>\n");
        for msg in &self.messages {
            history.push_str(&format!("{} - {}\n", msg.author, msg.text));
        }
        history.push_str("</История переписки>");
        history
    }
}

fn setup_logger() -> Result<(), fern::InitError> {
    let date = Local::now().format("%d-%m-%y").to_string();
    let log_dir = format!("logs/{}", date);

    fs::create_dir_all(&log_dir)?;

    let log_file = format!("{}/bot.log", log_dir);

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Info)
        .chain(std::io::stdout())
        .chain(fern::log_file(&log_file)?)
        .apply()?;

    Ok(())
}

async fn get_random_youtube_video(api_key: &str) -> Result<String, String> {
    let client = reqwest::Client::new();

    let search_terms = vec![
        "femboy",
        "инцел",
        "альт",
        "обзор",
        "алкоголь",
        ""
    ];

    let search_query = {
        let mut rng = rand::thread_rng();
        search_terms[rng.gen_range(0..search_terms.len())]
    };

    let url = format!(
        "https://www.googleapis.com/youtube/v3/search?part=snippet&q={}&type=video&maxResults=20&key={}",
        search_query, api_key
    );

    log::info!("Searching YouTube for: {}", search_query);

    let response = client.get(&url).send().await
        .map_err(|e| format!("Request error: {}", e))?;
    let data: YouTubeSearchResponse = response.json().await
        .map_err(|e| format!("JSON parse error: {}", e))?;

    if data.items.is_empty() {
        return Err("No videos found".into());
    }

    let random_index = {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..data.items.len())
    };

    let random_video = &data.items[random_index];

    let video_id = match &random_video.id {
        VideoId::Video { video_id } => video_id.clone(),
        VideoId::String(s) => s.clone(),
    };

    Ok(format!("https://www.youtube.com/watch?v={}", video_id))
}

async fn get_ai_response(api_key: &str, system_prompt: &str, user_message: &str, chat_history: &str) -> Result<String, String> {
    let client = reqwest::Client::new();

    // Формируем полный системный промпт с историей
    let full_system_prompt = format!("{}{}", system_prompt, chat_history);

    // Логируем промпт для отладки
    log::info!("System prompt: {}", full_system_prompt);
    log::info!("User message: {}", user_message);

    // Добавляем напоминание о стиле в user message для лучшего следования
    let enhanced_user_message = format!(
        "[Помни: пиши с ошибками, специально делай ошибки в словах, грамматические и пунктуационные ошибки ]\n\n{}",
        user_message
    );

    let request = GroqRequest {
        model: "llama-3.3-70b-versatile".to_string(),
        messages: vec![
            GroqMessage {
                role: "system".to_string(),
                content: full_system_prompt,
            },
            GroqMessage {
                role: "user".to_string(),
                content: enhanced_user_message,
            },
        ],
        max_tokens: 150,
        temperature: 0.4,     // Повысил для более креативных/хаотичных ответов
    };

    log::info!("Sending request to Groq API...");

    let response = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request error: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("API error {}: {}", status, error_text));
    }

    let data: GroqResponse = response.json().await
        .map_err(|e| format!("JSON parse error: {}", e))?;

    if data.choices.is_empty() {
        return Err("No response from AI".into());
    }

    Ok(data.choices[0].message.content.clone())
}

// Простая структура для парсинга сообщений без рекурсии
#[derive(Debug, Deserialize)]
struct SimpleMessage {
    message_id: i64,
    chat: SimpleChat,
    text: Option<String>,
    from: Option<SimpleUser>,
    date: i64,
    reply_to_message: Option<ReplyInfo>,  // Не рекурсивная структура
}

#[derive(Debug, Deserialize)]
struct SimpleChat {
    id: i64,
    #[serde(rename = "type")]
    chat_type: String,
    title: Option<String>,
    username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SimpleUser {
    id: i64,
    first_name: String,
    last_name: Option<String>,
    username: Option<String>,
    is_bot: Option<bool>,
}

// Плоская структура для reply - без вложенного reply_to_message
#[derive(Debug, Deserialize)]
struct ReplyInfo {
    message_id: i64,
    from: Option<SimpleUser>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<Value>, // Парсим как Value чтобы избежать рекурсии
}

#[derive(Debug, Deserialize)]
struct TelegramResponse {
    ok: bool,
    result: Vec<TelegramUpdate>,
}

async fn handle_dm_forward(
    bot: &Bot,
    msg: &SimpleMessage,
    allowed_chat_id: i64,
    groq_api_key: &str,
    groq_system_prompt: &str,
) {
    let user_id = msg.from.as_ref().map(|u| u.id).unwrap_or(0);
    let username = msg.from.as_ref().and_then(|u| u.username.clone()).unwrap_or_default();

    if let Some(text) = &msg.text {
        log::info!("DM Forward: Processing message from user {} (@{})", user_id, username);
        log::info!("DM Forward: Original text: {}", text);

        match get_ai_response(groq_api_key, groq_system_prompt, text, "").await {
            Ok(ai_response) => {
                log::info!("DM Forward: AI response received ({} chars)", ai_response.len());

                // Разбиваем по переносам строк
                let lines: Vec<&str> = ai_response
                    .split('\n')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                if lines.is_empty() {
                    log::warn!("DM Forward: AI response was empty");
                    let _ = bot.send_message(ChatId(msg.chat.id), "Не получилось обработать :(").await;
                    return;
                }

                // Отправляем каждую строку как отдельное сообщение в разрешённый чат
                for (i, line) in lines.iter().enumerate() {
                    let msg_text = if line.len() > 4000 {
                        format!("{}...", &line[..3997])
                    } else {
                        line.to_string()
                    };

                    if let Err(e) = bot.send_message(ChatId(allowed_chat_id), msg_text).await {
                        log::error!("DM Forward: Failed to send message {}: {}", i + 1, e);
                    }

                    // Задержка между сообщениями
                    if i < lines.len() - 1 {
                        let delay_ms = {
                            let mut rng = rand::thread_rng();
                            rng.gen_range(1000..=3000)
                        };
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }

                // Подтверждаем отправителю
                let _ = bot.send_message(ChatId(msg.chat.id), "Отправлено").await;
            }
            Err(e) => {
                log::error!("DM Forward: Failed to get AI response: {}", e);
                let _ = bot.send_message(ChatId(msg.chat.id), format!("Ошибка: {}", e)).await;
            }
        }
    }
}

async fn handle_message_simple(
    bot: &Bot,
    msg: SimpleMessage,
    allowed_chat_id: i64,
    allowed_user_ids: &[i64],
    groq_api_key: &str,
    groq_system_prompt: &str,
    bot_username: &str,
    chat_history: Arc<RwLock<ChatHistory>>,
) {
    let chat_id = msg.chat.id;

    // Обработка личных сообщений от разрешённых пользователей
    if msg.chat.chat_type == "private" {
        let user_id = msg.from.as_ref().map(|u| u.id).unwrap_or(0);

        if allowed_user_ids.contains(&user_id) {
            log::info!("DM from allowed user {}", user_id);
            handle_dm_forward(bot, &msg, allowed_chat_id, groq_api_key, groq_system_prompt).await;
        } else {
            log::info!("DM from unauthorized user {}", user_id);
        }
        return;
    }

    if chat_id != allowed_chat_id {
        log::info!("=== Message from unauthorized chat ===");
        log::info!("Chat ID: {}", chat_id);
        log::info!("Chat type: {}", msg.chat.chat_type);
        log::info!("Chat title: {:?}", msg.chat.title);
        log::info!("Chat username: {:?}", msg.chat.username);

        if let Some(user) = &msg.from {
            log::info!("From user ID: {}", user.id);
            log::info!("From username: {:?}", user.username);
            log::info!("From first name: {}", user.first_name);
            log::info!("From last name: {:?}", user.last_name);
        }

        if let Some(text) = &msg.text {
            log::info!("Message text: {}", text);
        }

        log::info!("Message ID: {}", msg.message_id);
        log::info!("Date: {}", msg.date);
        log::info!("=====================================");

        return;
    }

    log::info!("=== MAIN: Message from allowed chat ===");
    log::info!("MAIN: Chat ID: {}", chat_id);
    log::info!("MAIN: Chat type: {}", msg.chat.chat_type);
    log::info!("MAIN: Chat title: {:?}", msg.chat.title);
    log::info!("MAIN: Chat username: {:?}", msg.chat.username);

    if let Some(user) = &msg.from {
        log::info!("MAIN: From user ID: {}", user.id);
        log::info!("MAIN: From username: {:?}", user.username);
        log::info!("MAIN: From first name: {}", user.first_name);
        log::info!("MAIN: From last name: {:?}", user.last_name);
    }

    if let Some(text) = &msg.text {
        log::info!("MAIN: Message text: {}", text);

        // Определяем автора сообщения
        let author = msg.from.as_ref()
            .map(|u| u.username.clone().unwrap_or_else(|| u.first_name.clone()))
            .unwrap_or_else(|| "Unknown".to_string());

        // Сохраняем сообщение в историю
        {
            let mut history = chat_history.write().await;
            history.add_message(author.clone(), text.clone());
        }

        let random_chance_respond = {
            let mut rng = rand::thread_rng();
            rng.gen_range(1..=1000)
        };

        let random_chance_lot = {
            let mut rng = rand::thread_rng();
            rng.gen_range(1..=1000)
        };

        let bot_mention = format!("@{}", bot_username);
        let is_bot_mentioned = text.contains(&bot_mention);

        // Проверяем, это ответ на сообщение бота?
        let is_reply_to_bot = msg.reply_to_message.as_ref().map_or(false, |reply| {
            reply.from.as_ref().map_or(false, |from| {
                from.is_bot.unwrap_or(false) ||
                from.username.as_ref().map_or(false, |u| u == bot_username)
            })
        });

        if is_reply_to_bot {
            log::info!("MAIN: This is a reply to bot's message");
        }

        if random_chance_lot <= 5 {
            log::info!("MAIN: Triggered 0.5% chance event - asking for opinion");
            if let Err(e) = bot.send_message(ChatId(chat_id), "Мнение лота по этому вопросу? @ebokoshm").await {
                log::error!("Failed to send message: {}", e);
            }
        }

        // Отвечаем если: упомянули бота, ответили на сообщение бота, или 0.5% шанс
        let should_respond_with_ai = is_bot_mentioned || is_reply_to_bot || random_chance_respond <= 5;

        if should_respond_with_ai {
            log::info!("MAIN: AI response triggered (mentioned: {}, reply_to_bot: {}, random: {})",
                is_bot_mentioned, is_reply_to_bot, random_chance_respond);

            // Если это ответ на сообщение бота, добавляем контекст
            let context_message = if is_reply_to_bot {
                if let Some(reply) = &msg.reply_to_message {
                    if let Some(reply_text) = &reply.text {
                        format!("[Контекст - твоё предыдущее сообщение: {}]\n\nОтвет пользователя: {}", reply_text, text)
                    } else {
                        text.clone()
                    }
                } else {
                    text.clone()
                }
            } else {
                text.clone()
            };

            // Получаем историю для промпта и добавляем инфо об авторе
            let history_for_prompt = {
                let history = chat_history.read().await;
                let hist = history.format_for_prompt();
                format!("{}\n\n[Тебе пишет: {}]", hist, author)
            };

            match get_ai_response(groq_api_key, groq_system_prompt, &context_message, &history_for_prompt).await {
                Ok(ai_response) => {
                    log::info!("MAIN: AI response received ({} chars)", ai_response.len());

                    // Сохраняем ответ бота в историю
                    {
                        let mut history = chat_history.write().await;
                        history.add_message(bot_username.to_string(), ai_response.clone());
                    }

                    // Разбиваем ответ по строкам и отправляем каждую отдельно
                    let lines: Vec<&str> = ai_response
                        .split('\n')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();

                    if lines.is_empty() {
                        log::warn!("MAIN: AI response was empty after splitting");
                    } else if lines.len() == 1 {
                        // Одна строка — отправляем как обычно
                        let msg_text = if lines[0].len() > 4000 {
                            format!("{}...", &lines[0][..3997])
                        } else {
                            lines[0].to_string()
                        };
                        if let Err(e) = bot.send_message(ChatId(chat_id), msg_text).await {
                            log::error!("MAIN: Failed to send AI response: {}", e);
                        }
                    } else {
                        // Несколько строк — отправляем с задержкой
                        log::info!("MAIN: Sending {} messages with delays", lines.len());
                        for (i, line) in lines.iter().enumerate() {
                            let msg_text = if line.len() > 4000 {
                                format!("{}...", &line[..3997])
                            } else {
                                line.to_string()
                            };

                            if let Err(e) = bot.send_message(ChatId(chat_id), msg_text).await {
                                log::error!("MAIN: Failed to send message {}: {}", i + 1, e);
                            }

                            // Задержка 1-3 секунды между сообщениями (кроме последнего)
                            if i < lines.len() - 1 {
                                let delay_ms = {
                                    let mut rng = rand::thread_rng();
                                    rng.gen_range(1000..=3000)
                                };
                                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("MAIN: Failed to get AI response: {}", e);
                }
            }
        }
    }

    log::info!("MAIN: Message ID: {}", msg.message_id);
    log::info!("MAIN: Date: {}", msg.date);
    log::info!("MAIN: =====================================");
}

fn calculate_next_send_time() -> chrono::DateTime<Local> {
    let mut rng = rand::thread_rng();
    let now = Local::now();

    let random_hour = rng.gen_range(10..22);
    let random_minute = rng.gen_range(0..60);

    let target_time = NaiveTime::from_hms_opt(random_hour, random_minute, 0)
        .expect("Invalid time");

    let today_target = now.date_naive()
        .and_time(target_time)
        .and_local_timezone(Local)
        .unwrap();

    if today_target > now {
        today_target
    } else {
        (now + Duration::days(1))
            .date_naive()
            .and_time(target_time)
            .and_local_timezone(Local)
            .unwrap()
    }
}

async fn video_sender_task(bot: Bot, chat_id: i64, youtube_api_key: String) {
    loop {
        let next_send = calculate_next_send_time();
        log::info!("Next video will be sent at: {}", next_send.format("%Y-%m-%d %H:%M:%S"));

        let now = Local::now();
        let wait_duration = (next_send - now).to_std().unwrap_or(std::time::Duration::from_secs(60));

        tokio::time::sleep(wait_duration).await;

        log::info!("Attempting to send random YouTube video...");

        match get_random_youtube_video(&youtube_api_key).await {
            Ok(video_url) => {
                log::info!("Found video: {}", video_url);

                match bot.send_message(ChatId(chat_id), format!("Хуя че на ютубе!\n\n{}", video_url)).await {
                    Ok(_) => log::info!("Video sent successfully!"),
                    Err(e) => log::error!("Failed to send video: {}", e),
                }
            }
            Err(e) => {
                log::error!("Failed to get random video: {}", e);
            }
        }
    }
}

fn main() {
    // Увеличиваем размер стека для избежания stack overflow при обработке сложных JSON структур
    // Важно: нужно увеличить стек и для tokio worker потоков, т.к. десериализация происходит там
    // Message в teloxide очень большая структура, особенно с reply_to_message
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(32 * 1024 * 1024) // 32 MB для всех worker потоков
        .build()
        .expect("Failed to create tokio runtime");

    rt.block_on(async_main());
}

async fn async_main() {
    dotenvy::dotenv().ok();

    setup_logger().expect("Failed to initialize logger");

    log::info!("Starting telegram bot...");

    let bot = Bot::from_env();
    let token = env::var("TELOXIDE_TOKEN").expect("TELOXIDE_TOKEN must be set");

    let allowed_chat_id: i64 = env::var("ALLOWED_CHAT_ID")
        .expect("ALLOWED_CHAT_ID must be set")
        .parse()
        .expect("ALLOWED_CHAT_ID must be a valid number");

    let allowed_user_ids: Vec<i64> = env::var("ALLOWED_USER_IDS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    log::info!("Allowed user IDs for DM forwarding: {:?}", allowed_user_ids);

    let youtube_api_key = env::var("YOUTUBE_API_KEY")
        .expect("YOUTUBE_API_KEY must be set");

    let groq_api_key = env::var("GROQ_API_KEY")
        .expect("GROQ_API_KEY must be set");

    // Загружаем системный промпт из файла (более надёжно чем из .env)
    let groq_system_prompt = fs::read_to_string("system_prompt.txt")
        .unwrap_or_else(|_| {
            env::var("GROQ_SYSTEM_PROMPT")
                .unwrap_or_else(|_| "Ты норсолтон.".to_string())
        });

    log::info!("Loaded system prompt ({} chars)", groq_system_prompt.len());

    log::info!("Allowed chat ID: {}", allowed_chat_id);

    // Получаем информацию о боте для проверки упоминаний
    let bot_username = match bot.get_me().await {
        Ok(me) => {
            log::info!("Bot info: {:?}", me);
            me.username.clone().unwrap_or_else(|| {
                log::warn!("Bot has no username!");
                "bot".to_string()
            })
        }
        Err(e) => {
            log::error!("Failed to get bot info: {}", e);
            "bot".to_string()
        }
    };

    log::info!("Bot username: @{}", bot_username);

    // Инициализируем историю сообщений (последние 50)
    let chat_history = Arc::new(RwLock::new(ChatHistory::new(50)));
    log::info!("Chat history initialized (max 50 messages)");

    let bot_clone = bot.clone();
    tokio::spawn(async move {
        video_sender_task(bot_clone, allowed_chat_id, youtube_api_key).await;
    });

    log::info!("Next video will be sent at: {}", calculate_next_send_time().format("%Y-%m-%d %H:%M:%S"));

    // Ручной polling для избежания stack overflow
    let client = reqwest::Client::new();
    let mut offset: i64 = 0;

    // Сначала сбросим все старые апдейты
    log::info!("Dropping all pending updates...");
    let drop_url = format!(
        "https://api.telegram.org/bot{}/getUpdates?offset=-1&limit=1",
        token
    );
    if let Ok(resp) = client.get(&drop_url).send().await {
        if let Ok(data) = resp.json::<TelegramResponse>().await {
            if let Some(last_update) = data.result.last() {
                offset = last_update.update_id + 1;
                log::info!("Dropped pending updates, starting from offset {}", offset);
            }
        }
    }

    loop {
        let url = format!(
            "https://api.telegram.org/bot{}/getUpdates?offset={}&timeout=30",
            token, offset
        );

        match client.get(&url).send().await {
            Ok(response) => {
                match response.json::<TelegramResponse>().await {
                    Ok(data) => {
                        if !data.ok {
                            log::error!("Telegram API returned ok=false");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            continue;
                        }

                        for update in data.result {
                            offset = update.update_id + 1;

                            if let Some(msg_value) = update.message {
                                // Парсим сообщение в нашу простую структуру без рекурсии
                                match serde_json::from_value::<SimpleMessage>(msg_value) {
                                    Ok(msg) => {
                                        handle_message_simple(
                                            &bot,
                                            msg,
                                            allowed_chat_id,
                                            &allowed_user_ids,
                                            &groq_api_key,
                                            &groq_system_prompt,
                                            &bot_username,
                                            chat_history.clone(),
                                        ).await;
                                    }
                                    Err(e) => {
                                        log::debug!("Failed to parse message (probably has unsupported fields): {}", e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to parse Telegram response: {}", e);
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to fetch updates: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}
