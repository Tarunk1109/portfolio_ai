import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';
import fs from 'fs/promises';
import OpenAI from 'openai';
import 'dotenv/config';

// --- CONFIGURATION ---
const app = express();
const port = 3000;
const modelName = 'Xenova/all-MiniLM-L6-v2';
const embeddingsFile = 'tarunEmbeddings.json';

// --- STATE ---
let personalEmbeddings; 
let extractor;

// Initialize OpenAI client (it will automatically pick up the key from .env)
const openai = new OpenAI();


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

app.post('/chat', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
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
    if (!process.env.OPENAI_API_KEY) {
      // If no API key, return a simulated response for testing
      console.log("No OpenAI key found. Returning simulated response.");
      const simulatedResponse = `(Simulated Response) I found some relevant information for your question: "${question}".\n\nContext Found:\n${context}`;
      return res.json({ answer: simulatedResponse });
    }

    // 3. If API key exists, call OpenAI to generate a real answer
    const systemPrompt = `You are a helpful chatbot assistant for Tarun Karnati's personal portfolio. Your name is TK-Bot. Answer the user's question based *only* on the provided context. Be friendly, concise, and professional. If the context doesn't contain the answer, say that you don't have that information. Do not make things up. Here is the relevant context about Tarun:`;
    
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
  try {
    console.log('Loading AI model and personal data...');
    // Load the feature extraction model
    extractor = await pipeline('feature-extraction', modelName, { quantized: true });
    // Load your personal data embeddings from the file
    const embeddingsData = await fs.readFile(embeddingsFile, 'utf-8');
    personalEmbeddings = JSON.parse(embeddingsData);
    console.log('✅ Model and data loaded successfully.');
    
    app.listen(port, () => {
      console.log(`🚀 Server is running on http://localhost:${port}`);
    });
  } catch (error) {
    console.error('Failed to initialize server:', error);
  }
}

startServer();

