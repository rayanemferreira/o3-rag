import express from "express";
import axios from "axios";
import { ChromaClient } from "chromadb";
import multer from "multer";
import fs from "fs";
import ollama from 'ollama';

const EMB_MODEL = process.env.EMB_MODEL || "all-minilm";
const GEN_MODEL = process.env.GEN_MODEL || "llama3.2";
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434'; 
const COLLECTION_NAME = 'conversas';
const DIST_THRESHOLD = Number(process.env.DIST_THRESHOLD || '0.6');
const app = express();


app.use(express.json());
if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads", { recursive: true });
}


const upload = multer({ dest: "uploads/" });
const CHROMA_URL = 'http://localhost:8000';
const chroma = new ChromaClient({ path: CHROMA_URL });

let collection;
const collectionReady = (async () => {
  try {
    collection = await chroma.getOrCreateCollection({
      name: COLLECTION_NAME,
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: null, 
    });
    console.log(`Coleção '${COLLECTION_NAME}' pronta!`);
  } catch (err) {
    console.error("Falha ao conectar no ChromaDB:", err.message);
  }
})();


async function generateEmbedding(text) {
  const res = await ollama.embeddings({
    model: EMB_MODEL,
    prompt: text,
  }, { host: OLLAMA_HOST });
  return res.embedding;
}



app.post("/ia-prompt", async (req, res) => {
  const { text, threshold, topK } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    await collectionReady;
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");

    
    const queryEmbedding = await generateEmbedding(text);  
    console.log(`[ia-prompt] embedding dimensão=${Array.isArray(queryEmbedding) ? queryEmbedding.length : 'n/a'}`);
    const k = typeof topK === 'number' && topK > 0 ? Math.min(20, topK) : 5;
    const q = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: k,
      include: ["documents", "metadatas", "distances"],
    });

    const ids = (q.ids && q.ids[0]) || [];
    const docsRaw = ((q.documents && q.documents[0]) || []).map(String);
    const dists = (q.distances && q.distances[0]) || [];
    const thr = typeof threshold === 'number' ? threshold : DIST_THRESHOLD;
    const filtered = docsRaw.filter((_, i) => typeof dists[i] === 'number' ? dists[i] <= thr : true);
    const docs = Array.from(new Set(filtered));
    console.log(`[ia-prompt] hits=${ids.length} kept=${docs.length} threshold=${thr} k=${k} dists=${JSON.stringify(dists)}`);

    const context = docs
      .map((d) => `${d}`)
      .join("\n\n---\n\n");

    const hasContext = docs.length > 0;
    if (!hasContext) {
      return res.json({ ok: true, model: "llama3.2", answer: "Não encontrei essa informação no contexto." });
    }
    const prompt =
      (hasContext
        ? `Você é um assistente que responde de forma direta e concisa com base SOMENTE no CONTEXTO abaixo.\n` +
          `- Se não houver informação no contexto, responda exatamente: "Não encontrei essa informação no contexto."\n\n` +
          `CONTEXTO:\n${context}\n\n` +
          `PERGUNTA:\n${text}\n\n` +
          `RESPOSTA:`
        : `Não há contexto recuperado do banco. Responda apenas com base na pergunta, de forma objetiva.\n\n` +
          `${text}\n\n` +
          `RESPOSTA:`);

    
    const response = await axios.post(`${OLLAMA_HOST}/api/generate`, {
      model: GEN_MODEL,
      prompt,
      stream: false,
      options: { temperature: 0 }
    });

    
    const rawText = typeof response?.data?.response === "string" ? response.data.response : "";
    const answer = rawText
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .slice(0, 4)
      .join("\n");
    res.json({
      ok: true,
      model: response?.data?.model || GEN_MODEL,
      answer,
    });
  } catch (err) {
    console.error("Erro ao chamar Ollama:", err.message);
    res.status(500).send("Erro ao gerar resposta da IA.");
  }
});

app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).send("Arquivo é obrigatório");

  try {
    await collectionReady;
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");

    const content = fs.readFileSync(req.file.path, "utf-8");
    const lines = content.split("\n");

    let inseridos = 0;

    for (let i = 0; i < lines.length; i++) {
      const lineText = lines[i].trim();
      if (!lineText) continue;

      try {
        const emb = await generateEmbedding(lineText);

        const docId = Date.now().toString() + "_" + i;

        await collection.add({
          ids: [docId],
          documents: [lineText], 
          embeddings: [emb],
          metadatas: [{
            line_index: i,
            raw: lineText 
          }],
        });

        inseridos++;
        console.log(`Linha ${i} inserida com embedding (${emb.length} dimensões)`);

      } catch (err) {
        console.error(`Erro ao processar linha ${i}:`, err.message);
      }
    }

    res.json({ ok: true, inseridos });

  } catch (err) {
    console.error("Erro ao salvar txt:", err.message);
    res.status(500).send("Erro ao salvar txt no banco");
  } finally {
    fs.unlinkSync(req.file.path);
  }
});



app.get("/list", async (req, res) => {
  try {
    await collectionReady; 
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    const results = await collection.get({
      include: ["embeddings", "documents", "metadatas"],
    });
    res.json(results);
  } catch (err) {
    console.error("Erro ao listar:", err.message);
    res.status(500).send("Erro ao listar documentos");
  }
});




app.post("/search", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).send("Query é obrigatória");

  try {
    await collectionReady;
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    const queryEmb = await generateEmbedding(query);

    const results = await collection.query({
      queryEmbeddings: [queryEmb],
      nResults: 3,
    });

    res.json(results);
  } catch (err) {
    console.error("Erro ao buscar:", err.message);
    res.status(500).send("Erro ao buscar no banco");
  }
});


app.post("/ia", async (req, res) => {
  const { text, threshold, topK } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    await collectionReady;
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");

    const queryEmbedding = await generateEmbedding(text);
    console.log(`[ia] embedding dimensão=${Array.isArray(queryEmbedding) ? queryEmbedding.length : 'n/a'}`);

    const k = typeof topK === 'number' && topK > 0 ? Math.min(20, topK) : 5;
    const q = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: k,
      include: ["documents", "metadatas", "distances"],
    });

    const ids = (q.ids && q.ids[0]) || [];
    const docsRaw = ((q.documents && q.documents[0]) || []).map(String);
    const dists = (q.distances && q.distances[0]) || [];
    const thr = typeof threshold === 'number' ? threshold : DIST_THRESHOLD;
    const filtered = docsRaw.filter((_, i) => typeof dists[i] === 'number' ? dists[i] <= thr : true);
    const docs = Array.from(new Set(filtered));
    console.log(`[ia] hits=${ids.length} kept=${docs.length} threshold=${thr} k=${k} dists=${JSON.stringify(dists)}`);

    const context = docs
      .map((d) => `${d}`)
      .join("\n\n---\n\n");

    const hasContext = docs.length > 0;
    if (!hasContext) {
      return res.send("Não encontrei essa informação no contexto.");
    }
    const prompt =
      (hasContext
        ? `Você é um assistente que responde de forma direta e concisa com base SOMENTE no CONTEXTO abaixo.\n` +
          `- Se não houver informação no contexto, responda exatamente: "Não encontrei essa informação no contexto."\n\n` +
          `CONTEXTO:\n${context}\n\n` +
          `PERGUNTA:\n${text}\n\n` +
          `RESPOSTA:`
        : `Não há contexto recuperado do banco. Responda apenas com base na pergunta, de forma objetiva.\n\n` +
          `${text}\n\n` +
          `RESPOSTA:`);

    const response = await axios.post(`${OLLAMA_HOST}/api/generate`, {
      model: GEN_MODEL,
      prompt,
      stream: false,
      options: { temperature: 0 }
    });

    const rawText = typeof response?.data?.response === "string" ? response.data.response : "";
    const answer = rawText
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .slice(0, 4)
      .join("\n");

    res.send(answer);
  } catch (err) {
    console.error("Erro ao chamar Ollama:", err.message);
    res.status(500).send("Erro ao gerar resposta da IA.");
  }
});


app.listen(3000, () => {
  console.log("Servidor rodando na porta 3000");
});
