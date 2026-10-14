import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import 'dotenv/config';

// --- CONFIGURATION ---
const app = express();
const port = Number(process.env.PORT || 3000);
const modelName = 'Xenova/all-MiniLM-L6-v2';
const embeddingsFile = new URL('./tarunEmbeddings.json', import.meta.url);

// --- STATE ---
let personalEmbeddings; 
let extractor;

// Initialize OpenAI client (it will automatically pick up the key from .env)
const openai = process.env.OPENAI_API_KEY?.trim() ? new OpenAI() : null;


// --- UTILITY FUNCTIONS ---

// Function to calculate the cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Function to find the top N most relevant pieces of information
function findTopNMatches(queryEmbedding, dataEmbeddings, n = 3) {
  const similarities = dataEmbeddings.map(entry => ({
    source: entry.source,
    similarity: cosineSimilarity(queryEmbedding, entry.embedding)
  }));

  similarities.sort((a, b) => b.similarity - a.similarity);
  return similarities.slice(0, n);
}


// --- API ENDPOINT ---
app.use(cors());
app.use(express.json());

// Serve only public assets, never .env or other repository files.
app.get('/', (req, res) => res.sendFile(fileURLToPath(new URL('./index.html', import.meta.url))));
app.get('/index.html', (req, res) => res.sendFile(fileURLToPath(new URL('./index.html', import.meta.url))));
app.get('/resume.pdf', (req, res) => res.sendFile(fileURLToPath(new URL('./resume.pdf', import.meta.url))));
app.get('/health', (req, res) => res.json({
  status: 'ok',
  chatbot: extractor && personalEmbeddings ? 'ready' : 'unavailable',
}));

app.post('/chat', async (req, res) => {
  const { question } = req.body || {};

  if (typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ error: 'Question is required' });
  }

  if (!extractor || !personalEmbeddings) {
    return res.status(503).json({ error: 'Chatbot is loading or unavailable. Please try again later.' });
  }

  try {
    // 1. Generate embedding for the user's question
    const queryEmbeddingOutput = await extractor(question, { pooling: 'mean', normalize: true });
    const queryEmbedding = Array.from(queryEmbeddingOutput.data);

    // 2. Find the most relevant context from your personal data
    const topMatches = findTopNMatches(queryEmbedding, personalEmbeddings, 3);
    const context = topMatches.map(match => match.source).join('\n\n---\n\n');

    // --- CHATBOT LOGIC ---

    // Check if the OpenAI API key is available
    if (!openai) {
      // If no API key, return a simulated response for testing
      console.log("No OpenAI key found. Returning simulated response.");
      const simulatedResponse = `(Simulated Response) I found some relevant information for your question: "${question}".\n\nContext Found:\n${context}`;
      return res.json({ answer: simulatedResponse });
    }

    // 3. If API key exists, call OpenAI to generate a real answer
    const systemPrompt = `You are a friendly and professional AI assistant named TK-Bot for Tarun Karnati's portfolio. Use the provided context to answer questions about him. Use only facts in the provided context. If a detail is unavailable, say that it is not documented. Preserve project status and distinguish simulated data, prototypes, academic work, and sandbox transactions from production use. Do not invent metrics or employment. Be concise, friendly, and professional.`;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini", 
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}` }
      ],
    });

    const answer = completion.choices[0].message.content;
    res.json({ answer });

  } catch (error) {
    console.error('Error processing chat request:', error);
    res.status(500).json({ error: 'Failed to process chat request' });
  }
});


// --- SERVER INITIALIZATION ---
async function startServer() {
  // Keep the portfolio available even while the model downloads or fails.
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  }).on('error', error => {
    console.error('Failed to listen:', error);
    process.exit(1);
  });
  try {
    console.log('Loading AI model and personal data...');
    // Load the feature extraction model
    extractor = await pipeline('feature-extraction', modelName, { quantized: true });
    // Load your personal data embeddings from the file
    const embeddingsData = await fs.readFile(embeddingsFile, 'utf-8');
    personalEmbeddings = JSON.parse(embeddingsData);
    console.log('✅ Model and data loaded successfully.');
    

  } catch (error) {
    console.error('Failed to initialize server:', error);
  }
}

startServer();

