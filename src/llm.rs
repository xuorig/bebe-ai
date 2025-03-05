use serde::{Deserialize, Serialize};

pub async fn chat(gemini_key: &str, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let gemini_client = reqwest::Client::new();

    let gemini_generate_url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}", gemini_key);

    let gemini_request = GeminiRequest::from_prompt(&prompt);

    tracing::info!("Asking gemini...");

    let gemini_response = gemini_client
        .post(gemini_generate_url)
        .header("content-type", "application/json")
        .json(&gemini_request)
        .send()
        .await?
        .json::<GeminiResponse>()
        .await?;

    let response = gemini_response.candidates[0].content.parts[0].text.clone();

    Ok(response)
}

#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    system_instruction: GeminiSystemInstruction,
    contents: Vec<GeminiContent>,
}

impl GeminiRequest {
    pub fn from_prompt(prompt: &str) -> Self {
        let parts = vec![GeminiContent {
            parts: vec![GeminiPart {
                text: prompt.to_string(),
            }],
        }];

        GeminiRequest {
            system_instruction: GeminiSystemInstruction {
                parts: vec![GeminiPart {
                    text: "You are an helpful AI assistant that answers questions about newborn and pregnancy health. Only use information provided in the prompt, never use external knowledge. Feel free to quote the provided context always. Always answer in the language the question is in.".to_string(),
                }]
            },
            contents: parts,
        }
    }
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeminiContent {
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeminiPart {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiContent,
}
