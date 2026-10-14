# Tarun Karnati — Portfolio

An interactive, profile-first portfolio with a static frontend and optional Node.js retrieval backend.

## Local preview

```sh
npm ci
npm start
```

Open http://localhost:3000. The portfolio opens immediately; the chatbot model loads in the background. The first load downloads its embedding model.

Copy `.env.example` to `.env` to configure `PORT` or an optional `OPENAI_API_KEY`. With no API key, the chatbot returns matching context rather than generated answers. Keep `.env` private.

## Project content

- `index.html`: profile, visible project cards, experience, education, awards, and embedded project data.
- `portfolio.css`: responsive themes, animation, and component styling.
- `portfolio.js`: featured architecture, filters, keyboard search, case studies, and project comparison.
- `projects.json`: documented project inventory and public links.
- `tarunData.json`: chatbot knowledge, including project scope and status.
- `tarunEmbeddings.json`: generated retrieval vectors.

Keep the cards, embedded project data, inventory, and optional chatbot knowledge in sync when editing project content. Architecture descriptions are maintained in `portfolio.js`. Regenerate vectors after editing the knowledge:

```sh
npm run generate-embeddings
```

Restart the backend to load the new vectors.

## Hosting

GitHub Pages publishes the static frontend from the main branch at https://tarunk1109.github.io/portfolio_ai/. No custom domain is required. All portfolio interactions run locally in the browser and require no API key or hosted backend. The optional retrieval backend retains `/chat` for local development; it is not part of the public portfolio interface.

Only public source and demo links are displayed. Private repositories remain unlinked. Project scopes distinguish academic work, prototypes, synthetic data, and sandbox transactions from production use.
