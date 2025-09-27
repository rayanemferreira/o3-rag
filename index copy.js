import express from "express";
import axios from "axios";
import { ChromaClient } from "chromadb";
import multer from "multer";
import fs from "fs";

const app = express();
app.use(express.json());

// Config upload (destino temporário)
const upload = multer({ dest: "uploads/" });

// conecta no servidor Chroma
const chroma = new ChromaClient({ host: "localhost", port: 8000 });

// garante que a coleção existe
let collection;
(async () => {
  collection = await chroma.getOrCreateCollection({
    name: "conversas",
    embeddingFunction: null,
  });
  console.log("Coleção 'conversas' pronta!");
})();

// --- helper: quebra texto em pedaços ---
function chunkText(text, chunkSize = 500, overlap = 50) {
  const words = text.split(/\s+/);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    chunks.push(chunk);
    i += chunkSize - overlap;
  }
  return chunks;
}

// rota principal que chama o Ollama
app.post("/ia", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    // chama o Ollama
    const response = await axios.post("http://127.0.0.1:11434/api/generate", {
      model: "llama3.2",
      prompt: text,
      stream: false,
    });

    const respData = response.data.response.toString();

    // salva pergunta e resposta no Chroma
    const fakeEmbedding = [Math.random(), Math.random(), Math.random()];
    const docId = Date.now().toString();
    await collection.add({
      ids: [docId],
      documents: [`Pergunta: ${text} | Resposta: ${respData}`],
      embeddings: [fakeEmbedding],
    });

    res.send(respData);
  } catch (err) {
    console.error("Erro ao chamar Ollama:", err.message);
    res.status(500).send("Erro ao gerar resposta da IA.");
  }
});

// rota para upload de arquivo .txt
app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).send("Arquivo é obrigatório");

  try {
    // ler o txt
    const content = fs.readFileSync(req.file.path, "utf-8");

    // quebrar em pedaços
    const chunks = chunkText(content);

    const docs = [];
    const ids = [];
    const embeddings = [];

    chunks.forEach((c, idx) => {
      ids.push(Date.now().toString() + "_" + idx);
      docs.push(c);
      embeddings.push([Math.random(), Math.random(), Math.random()]); // fake
    });

    // salvar no chroma
    await collection.add({
      ids,
      documents: docs,
      embeddings,
    });

    res.json({ ok: true, chunks: chunks.length });
  } catch (err) {
    console.error("Erro ao salvar txt:", err.message);
    res.status(500).send("Erro ao salvar txt no banco");
  } finally {
    // remove o arquivo temporário
    fs.unlinkSync(req.file.path);
  }
});

// rota para listar todos documentos
app.get("/list", async (req, res) => {
  const results = await collection.get();
  res.json(results);
});

// rota para buscar (fake embedding)
app.post("/search", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).send("Query é obrigatória");

  const fakeEmbedding = [1, 2, 3]; // só para teste

  const results = await collection.query({
    queryEmbeddings: [fakeEmbedding],
    nResults: 3,
  });

  res.json(results);
});

app.listen(3000, () => {
  console.log("Servidor rodando na porta 3000 🚀");
});
