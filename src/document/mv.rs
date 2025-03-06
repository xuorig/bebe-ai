use std::sync::Arc;

use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use super::{Chunk, DocumentFetcher};

const ROOT: &str = "https://www.inspq.qc.ca/mieux-vivre/consultez-le-guide";
const BASE_URL: &str = "https://www.inspq.qc.ca";
const TEXT_MIN_LENGTH: usize = 42;

pub struct MieuxVivreFetcher {}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct MieuxVivreMetadata {
    pub title: String,
    pub section: String,
    pub subsection: String,
    pub heading: Option<String>,
    pub url: String,
}

#[derive(Debug)]
struct MVPageMetadata {
    section: String,
    subsection: Option<String>,
    url: String,
}

impl MieuxVivreFetcher {
    pub fn new() -> Self {
        Self {}
    }

    async fn get_page_content(
        client: &reqwest::Client,
        url: &str,
        section: &str,
        subsection: &str,
        semaphore: Arc<Semaphore>,
    ) -> Result<Vec<Chunk<MieuxVivreMetadata>>, reqwest::Error> {
        let _permit = semaphore.acquire().await;

        tracing::info!("Fetching {}", url);

        let mut chunks = vec![];

        let response = client.get(url).send().await?;
        let document = Html::parse_document(&response.text().await?);
        let selector = Selector::parse(".two-column-layout__left .field__item").unwrap();
        let title_selector = Selector::parse("h1").unwrap();

        let title = document.select(&title_selector).next().unwrap();
        let title = title.text().collect::<String>();

        let content = document.select(&selector).next().unwrap();

        let mut current_heading = None;

        content.child_elements().for_each(|element| {
            // Mieux Vivre uses h2 for headings within a page. We use this to determine the current heading.
            // Most of the elements within the main content are p tags. Sometimes,
            // a sub div is used for things like call outs. We grab the text from those sub divs as a chunk.

            // When we find a UL, we add it to the previous chunk.

            if element.value().name() == "h2"
                || element.value().name() == "h3"
                || element.value().name() == "h4"
            {
                current_heading = Some(element.text().collect::<String>());
            } else if element.value().name() == "p" {
                let text = element.text().collect::<String>();
                chunks.push(Chunk {
                    text,
                    metadata: MieuxVivreMetadata {
                        title: title.clone(),
                        section: section.to_string(),
                        subsection: subsection.to_string(),
                        heading: current_heading.clone(),
                        url: url.to_string(),
                    },
                });
            } else if element.value().name() == "div" || element.value().name() == "article" {
                current_heading = None;
                let text = element.text().collect::<String>();
                chunks.push(Chunk {
                    text,
                    metadata: MieuxVivreMetadata {
                        title: title.clone(),
                        section: section.to_string(),
                        subsection: subsection.to_string(),
                        heading: current_heading.clone(),
                        url: url.to_string(),
                    },
                });
            } else if element.value().name() == "ul" || element.value().name() == "ol" {
                let text = element.text().collect::<String>();
                match chunks.last_mut() {
                    Some(last_chunk) => {
                        last_chunk.text.push_str("\n");
                        last_chunk.text.push_str(&text);
                    }
                    None => {
                        chunks.push(Chunk {
                            text,
                            metadata: MieuxVivreMetadata {
                                title: title.clone(),
                                section: section.to_string(),
                                subsection: subsection.to_string(),
                                heading: current_heading.clone(),
                                url: url.to_string(),
                            },
                        });
                    }
                }
            } else {
                tracing::warn!("Unknown element: {:?}", element.value().name());
            }
        });

        Ok(chunks)
    }
}

impl DocumentFetcher<MieuxVivreMetadata> for MieuxVivreFetcher {
    async fn fetch(&self) -> Result<Vec<Chunk<MieuxVivreMetadata>>, Box<dyn std::error::Error>> {
        tracing::info!("Crawling and chunking Mieux Vivre");

        let client = reqwest::Client::builder().build()?;

        let response = client.get(ROOT).send().await?.text().await?;

        let document = Html::parse_document(&response);

        let selector = Selector::parse(".carte-lien-mv").unwrap();

        // Fetch all main sections pages
        let sections = document.select(&selector).map(|element| {
            element
                .first_child()
                .map(|element| {
                    let href = element
                        .value()
                        .as_element()
                        .map(|element| element.attr("href").unwrap())
                        .unwrap();

                    let title = element
                        .children()
                        .skip(1)
                        .next()
                        .map(|element| {
                            element
                                .first_child()
                                .map(|element| element.value().as_text().unwrap().text.to_string())
                                .unwrap()
                        })
                        .unwrap();

                    (title, href)
                })
                .expect("Failed to parse section link")
        });

        tracing::debug!("Found sections: {:#?}", sections);

        let mut pages = vec![];

        for (section_title, href) in sections {
            tracing::info!("Crawling {}", href);

            let response = client
                .get(format!("{}{}", BASE_URL, href))
                .send()
                .await?
                .text()
                .await?;

            let document = Html::parse_document(&response);
            let selector = Selector::parse("#block-mieuxvivre-post-content-menu .menu").unwrap();
            let ul = document.select(&selector).next().unwrap();
            ul.child_elements().for_each(|element| {
                let (subsection_title, href) = element
                    .child_elements()
                    .find(|e| e.value().name() == "a")
                    .map(|element| {
                        let href = element.value().attr("href").unwrap();
                        let title = element.text().collect::<String>();
                        (title, href.to_string())
                    })
                    .unwrap();

                pages.push(MVPageMetadata {
                    section: section_title.clone(),
                    subsection: Some(subsection_title.clone()),
                    url: format!("{}{}", BASE_URL, href),
                });

                element
                    .child_elements()
                    .find(|e| e.value().name() == "ul")
                    .map(|element| {
                        element.child_elements().for_each(|element| {
                            element
                                .child_elements()
                                .find(|e| e.value().name() == "a")
                                .map(|element| {
                                    let href = element.value().attr("href").unwrap();
                                    let page_title = element.text().collect::<String>();
                                    let url = format!("{}{}", BASE_URL, href);

                                    pages.push(MVPageMetadata {
                                        section: section_title.clone(),
                                        subsection: Some(subsection_title.clone()),
                                        url,
                                    });
                                });
                        });
                    });
            });
        }

        tracing::debug!("Found pages: {:#?}", pages);

        let mut chunks = vec![];

        let mut set = tokio::task::JoinSet::new();

        let semaphore = Arc::new(tokio::sync::Semaphore::new(25));

        for page in pages {
            let client = client.clone();
            let section = page.section.clone();
            let subsection = page.subsection.clone().unwrap_or_else(|| "".to_string());
            let url = page.url.clone();
            let semaphore = semaphore.clone();

            set.spawn(async move {
                MieuxVivreFetcher::get_page_content(&client, &url, &section, &subsection, semaphore)
                    .await
            });
        }

        while let Some(res) = set.join_next().await {
            let page_chunks = res.unwrap().unwrap();
            let filtered = page_chunks
                .into_iter()
                .filter(|chunk| chunk.text.len() > TEXT_MIN_LENGTH);
            chunks.extend(filtered);
        }

        Ok(chunks)
    }
}
