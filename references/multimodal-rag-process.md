Google Embedding Models

1. gemini-embedding-001 (only for texts)
2. gemini-embedding-2-preview (has multimodal embedding capability - works with images also)

# What should we do when docling finds different content types of PDF ?

1.  text
    extract the text and make it chunks in ingestion layer
2.  table
    extract the table -> convert to data frame -> make it chunks in ingestion layer
3.  image
    extract the image -> (1. convert the image to base64 img ) -> pass it to get saved in vector db
    [or]
    -> (2. save that image into s3 kind of cloud storage) -> save the url in vector db.
    ----

                      -> to further understand either the base64 image or s3 url -we need LLM (vision capable modal)  and ask it to give description of that image and then save the description in vector db


