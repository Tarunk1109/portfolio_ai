import { pipeline } from '@xenova/transformers';
import fs from 'fs/promises';

// --- CONFIGURATION ---
const inputFile = 'tarunData.json';
const outputFile = 'tarunEmbeddings.json';
const modelName = 'Xenova/all-MiniLM-L6-v2';

// --- MAIN FUNCTION ---
async function generateEmbeddings() {
  try {
    console.log('Step 1: Loading the AI model...');
    const extractor = await pipeline('feature-extraction', modelName, { quantized: true });
    console.log('Model loaded successfully.');

    console.log(`Step 2: Reading your personal data from ${inputFile}...`);
    const dataString = await fs.readFile(inputFile, 'utf-8');
    const data = JSON.parse(dataString);
    console.log('Data loaded.');

    console.log('Step 3: Processing and embedding your data...');
    const processedEntries = [];
    let entryCount = 0;

    // A helper function to process different data structures (strings, arrays, objects)
    const processEntry = async (key, value) => {
        let textToEmbed;
        if (typeof value === 'string') {
            textToEmbed = `${key}: ${value}`;
        } else if (Array.isArray(value)) {
            // Join array items into a coherent string
            textToEmbed = `${key}: ${value.join(', ')}`;
        } else if (typeof value === 'object' && value !== null) {
            // Convert simple objects to string format
            textToEmbed = `${key}: ${Object.entries(value).map(([k, v]) => `${k} - ${v}`).join('; ')}`;
        } else {
            return; // Skip non-textual data
        }

        // Generate the embedding for the text
        const output = await extractor(textToEmbed, { pooling: 'mean', normalize: true });
        const embedding = Array.from(output.data);

        processedEntries.push({
            source: textToEmbed,
            embedding: embedding,
        });
        entryCount++;
    };

    // Iterate over all top-level keys in the JSON data
    for (const key in data) {
        if (Object.hasOwnProperty.call(data, key)) {
            const value = data[key];
            if (Array.isArray(value)) {
                 // Special handling for arrays of objects (like projects, education)
                for(const item of value) {
                    await processEntry(key, item);
                }
            } else {
                await processEntry(key, value);
            }
        }
    }
    
    console.log(`Successfully generated embeddings for ${entryCount} data entries.`);

    console.log(`Step 4: Saving embeddings to ${outputFile}...`);
    await fs.writeFile(outputFile, JSON.stringify(processedEntries, null, 2));
    console.log('✅ All done! Your embeddings file has been created.');

  } catch (error) {
    console.error('An error occurred:', error);
  }
}

generateEmbeddings();

