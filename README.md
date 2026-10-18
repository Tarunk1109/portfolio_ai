# Tarun Karnati — Portfolio

A readable single-page portfolio with a static frontend and optional Node.js retrieval backend.

## Local preview

```sh
npm ci
npm start
```

Open http://localhost:3000. The portfolio opens immediately; the chatbot model loads in the background. The first load downloads its embedding model.

Copy `.env.example` to `.env` to configure `PORT` or an optional `OPENAI_API_KEY`. With no API key, the chatbot returns matching context rather than generated answers. Keep `.env` private.

## Project content

- `index.html`: profile, skills, 17 visible project descriptions, inline implementation details, experience, education, awards, and contact links.
- `portfolio.css`: responsive layout, readable typography, and light/dark colour themes.
- `portfolio.js`: category filters, project search, theme switching, mobile navigation, and active navigation links.
- `projects.json`: documented project inventory and public links.
- `tarunData.json`: chatbot knowledge, including project scope and status.
- `tarunEmbeddings.json`: generated retrieval vectors.

Keep the HTML project cards, inventory, and optional chatbot knowledge in sync when editing project content. Core information is available without dialogs or JavaScript. Regenerate vectors after editing the knowledge:

```sh
npm run generate-embeddings
```

Restart the backend to load the new vectors.

## Hosting

GitHub Pages publishes the static frontend from the main branch at https://tarunk1109.github.io/portfolio_ai/. No custom domain is required. All portfolio interactions run locally in the browser and require no API key or hosted backend. The optional retrieval backend retains `/chat` for local development; it is not part of the public portfolio interface.

Only public source and demo links are displayed. Private repositories remain unlinked. Project scopes distinguish academic work, prototypes, synthetic data, and sandbox transactions from production use.
