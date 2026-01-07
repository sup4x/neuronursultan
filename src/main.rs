use teloxide::prelude::*;
use chrono::{Local, Duration, NaiveTime};
use rand::Rng;
use serde::Deserialize;
use std::env;
use std::fs;

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

                match bot.send_message(ChatId(chat_id), format!("🎬 Случайное видео дня!\n\n{}", video_url)).await {
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

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    setup_logger().expect("Failed to initialize logger");

    log::info!("Starting telegram bot...");

    let bot = Bot::from_env();

    let allowed_chat_id: i64 = env::var("ALLOWED_CHAT_ID")
        .expect("ALLOWED_CHAT_ID must be set")
        .parse()
        .expect("ALLOWED_CHAT_ID must be a valid number");

    let youtube_api_key = env::var("YOUTUBE_API_KEY")
        .expect("YOUTUBE_API_KEY must be set");

    log::info!("Allowed chat ID: {}", allowed_chat_id);

    let bot_clone = bot.clone();
    tokio::spawn(async move {
        video_sender_task(bot_clone, allowed_chat_id, youtube_api_key).await;
    });

    teloxide::repl(bot, move |bot: Bot, msg: Message| async move {
        let chat_id = msg.chat.id.0;

        if chat_id != allowed_chat_id {
            log::info!("=== Message from unauthorized chat ===");
            log::info!("Chat ID: {}", chat_id);
            log::info!("Chat type: {:?}", msg.chat.kind);
            log::info!("Chat title: {:?}", msg.chat.title());
            log::info!("Chat username: {:?}", msg.chat.username());

            if let Some(user) = &msg.from {
                log::info!("From user ID: {}", user.id);
                log::info!("From username: {:?}", user.username);
                log::info!("From first name: {}", user.first_name);
                log::info!("From last name: {:?}", user.last_name);
            }

            if let Some(text) = msg.text() {
                log::info!("Message text: {}", text);
            }

            log::info!("Message ID: {}", msg.id);
            log::info!("Date: {:?}", msg.date);
            log::info!("=====================================");

            return Ok(());
        }

        log::info!("=== MAIN: Message from allowed chat ===");
        log::info!("MAIN: Chat ID: {}", chat_id);
        log::info!("MAIN: Chat type: {:?}", msg.chat.kind);
        log::info!("MAIN: Chat title: {:?}", msg.chat.title());
        log::info!("MAIN: Chat username: {:?}", msg.chat.username());

        if let Some(user) = &msg.from {
            log::info!("MAIN: From user ID: {}", user.id);
            log::info!("MAIN: From username: {:?}", user.username);
            log::info!("MAIN: From first name: {}", user.first_name);
            log::info!("MAIN: From last name: {:?}", user.last_name);
        }

        if let Some(text) = msg.text() {
            log::info!("MAIN: Message text: {}", text);

            // Вычисляем случайное значение до всех .await операций
            let random_value = {
                let mut rng = rand::thread_rng();
                rng.gen_range(1..=100)
            };

            // if text == "/start" {
            //     bot.send_message(
            //         msg.chat.id,
            //         "Привет! Я простой бот на Rust. Используй /start для начала работы."
            //     )
            //     .await?;
            // } else {
            //     bot.send_message(
            //         msg.chat.id,
            //         format!("Ты написал: {}", text)
            //     )
            //     .await?;
            // }

            // 5% шанс спросить мнение лота
            if random_value <= 5 {
                log::info!("MAIN: Triggered 5% chance event - asking for opinion");
                bot.send_message(
                    msg.chat.id,
                    "Мнение лота по этому вопросу? @ebokoshm"
                )
                .await?;
            }
        }

        log::info!("MAIN: Message ID: {}", msg.id);
        log::info!("MAIN: Date: {:?}", msg.date);
        log::info!("MAIN: =====================================");

        Ok(())
    })
    .await;
}
