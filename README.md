# Tarun Karnati — Portfolio

Static portfolio and optional Node.js retrieval chatbot.

## Local preview

```sh
npm ci
npm start
```

Open http://localhost:3000. The portfolio opens immediately; the chatbot model loads in the background. The first load downloads its embedding model.

Copy `.env.example` to `.env` to configure `PORT` or an optional `OPENAI_API_KEY`. With no API key, the chatbot returns matching context rather than generated answers. Keep `.env` private.

## Project content

- `index.html`: rendered project cards, filters, search, details, and award entries.
- `projects.json`: documented project inventory and public links.
- `tarunData.json`: chatbot knowledge, including project scope and status.
- `tarunEmbeddings.json`: generated retrieval vectors.

Keep the cards, inventory, and chatbot knowledge in sync when editing project content. Regenerate vectors after editing the knowledge:

```sh
npm run generate-embeddings
```

Restart the backend to load the new vectors.

## Hosting

GitHub Pages publishes the static frontend from the main branch at https://tarunk1109.github.io/portfolio_ai/. No custom domain is required. Pages does not run Node.js: the public chatbot calls the separate Render backend configured in `index.html`. Local previews call the local `/chat` endpoint.

Only public source and demo links are displayed. Private repositories remain unlinked. Project scopes distinguish academic work, prototypes, synthetic data, and sandbox transactions from production use.
