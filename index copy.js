import express from "express";
import axios from "axios";
import { ChromaClient } from "chromadb";
import multer from "multer";
import fs from "fs";
import ollama from 'ollama';

 
 
// modelo de embedding
const EMB_MODEL = "all-minilm";
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434'; // ajuste se necessário
const COLLECTION_NAME = 'conversas';

const app = express();
app.use(express.json());

// garante diretório de uploads
if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads", { recursive: true });
}

// Config upload (destino temporário)
const upload = multer({ dest: "uploads/" });

// conecta no servidor Chroma
const CHROMA_URL = 'http://localhost:8000';
const chroma = new ChromaClient({ path: CHROMA_URL });

// garante que a coleção existe
let collection;
const collectionReady = (async () => {
  try {
    collection = await chroma.getOrCreateCollection({
      name: COLLECTION_NAME,
      metadata: { "hnsw:space": "cosine" },
      embeddingFunction: null, // vamos gerar manualmente
    });
    console.log(`Coleção '${COLLECTION_NAME}' pronta!`);
  } catch (err) {
    console.error("Falha ao conectar no ChromaDB:", err.message);
  }
})();


 
 
// --- helper: parse de linha do WhatsApp ---
function parseLine(line) {
  const regex = /^(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}) - ([^:]+): (.*)$/;
  const m = line.match(regex);
  if (!m) return null;
  const [_, datePart, timePart, phonePart, msgPart] = m;
  const [dd, mm, yyyy] = datePart.split("/");
  const iso = `${yyyy}-${mm}-${dd}T${timePart}:00`;
  return {
    datetime: iso,
    phone: phonePart.trim(),
    message: msgPart.trim(),
    raw: line,
  };
}

// --- helper: gerar embedding real ---
async function generateEmbedding(text) {
  const res = await ollama.embeddings({
    model: EMB_MODEL,
    prompt: text,
  }, { host: OLLAMA_HOST });
  return res.embedding;
}



// rota principal que chama o Ollama
app.post("/ia-prompt", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady;

    // 1) gera embedding da pergunta
    const queryEmbedding = await generateEmbedding(text);

    // 2) busca top-K documentos similares
    const topK = 5;
    const q = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: topK,
      include: ["documents", "metadatas", "distances"],
    });

    const ids = (q.ids && q.ids[0]) || [];
    const docs = (q.documents && q.documents[0]) || [];
    const metas = (q.metadatas && q.metadatas[0]) || [];
    const dists = (q.distances && q.distances[0]) || [];

    const context = docs
      .map((d, i) => {
        const meta = metas[i] || {};
        const dist = typeof dists[i] === 'number' ? dists[i].toFixed(4) : dists[i];
        return `Trecho ${i + 1} (id=${ids[i] || ''}, distancia=${dist})\n` +
               `Data/hora: ${meta.datetime || ''} | Telefone: ${meta.phone || ''}\n` +
               `${d}`;
      })
      .join("\n\n---\n\n");

    const hasContext = docs.length > 0;
    const prompt = `Você é um assistente especialista. Responda em português de forma objetiva, clara e curta.\n` +
      `RESTRIÇÃO: sua resposta deve ter no máximo 4 linhas, sem listas longas, sem floreios.\n` +
      `Se não houver informação suficiente, diga isso em até 2 linhas.\n\n` +
      (hasContext
        ? `Use APENAS as informações a seguir como contexto (não invente):\n\n${context}\n\n`
        : `Não há contexto recuperado do banco. Responda apenas com base na pergunta.\n\n`) +
      `Pergunta do usuário: ${text}\n\n` +
      `Se a resposta não estiver claramente no contexto, diga que não foi possível encontrar com base nos dados.`;

    // 3) chama o modelo no Ollama com prompt contextualizado
    const response = await axios.post(`${OLLAMA_HOST}/api/generate`, {
      model: "llama3.2",
      prompt,
      stream: false,
    });

    // Ollama retorna o texto em data.response (não data.response.answer)
    const rawText = typeof response?.data?.response === "string" ? response.data.response : "";
    // Pós-processamento: manter no máximo 4 linhas não vazias
    const answer = rawText
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .slice(0, 4)
      .join("\n");
    res.json({
      ok: true,
      model: response?.data?.model || "llama3.2",
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
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady; // garante que a coleção existe
    const content = fs.readFileSync(req.file.path, "utf-8");
    const lines = content.split("\n");

    let inseridos = 0;

    for (let i = 0; i < lines.length; i++) {
      const parsed = parseLine(lines[i].trim());
      if (!parsed) continue;

      try {
        // gera embedding da linha
        const emb = await generateEmbedding(parsed.message);

        const docId = Date.now().toString() + "_" + i;

        // insere imediatamente no Chroma
        await collection.add({
          ids: [docId],
          documents: [parsed.message],
          embeddings: [emb],
          metadatas: [{
            phone: parsed.phone,
            datetime: parsed.datetime,
            raw: parsed.raw,
            line_index: i,
          }],
        });

        inseridos++;
        console.log(`✅ Linha ${i} inserida com embedding (${emb.length} dimensões)`);

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

 


// // rota para limpar a coleção (apagar todos documentos)
// app.delete("/clear", async (req, res) => {
//   try {
//     await collection.delete({ ids: [] }); // se não passar ids, limpa tudo
//     res.json({ ok: true, msg: "Coleção 'conversas' limpa com sucesso!" });
//   } catch (err) {
//     console.error("Erro ao limpar coleção:", err.message);
//     res.status(500).send("Erro ao limpar a coleção.");
//   }
// });

// rota para listar todos documentos
app.get("/list", async (req, res) => {
  try {
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady; // garante que a coleção existe
    const results = await collection.get({
      include: ["embeddings", "documents", "metadatas"],
    });
    res.json(results);
  } catch (err) {
    console.error("Erro ao listar:", err.message);
    res.status(500).send("Erro ao listar documentos");
  }
});

// rota para buscar um subconjunto com embeddings garantidos
app.post("/list_with_embeddings", async (req, res) => {
  try {
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady;
    const { ids, limit } = req.body || {};
    const options = { include: ["embeddings", "documents", "metadatas"] };
    if (Array.isArray(ids) && ids.length > 0) {
      options.ids = ids;
    }
    if (typeof limit === 'number' && limit > 0) {
      options.limit = limit;
    }
    const results = await collection.get(options);
    res.json(results);
  } catch (err) {
    console.error("Erro ao listar com embeddings:", err.message);
    res.status(500).send("Erro ao listar com embeddings");
  }
});

// rota para buscar
app.post("/search", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).send("Query é obrigatória");

  try {
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady; // garante que a coleção existe
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
  const { text } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    if (!collection) return res.status(503).send("ChromaDB indisponível no momento.");
    await collectionReady;

    // Pega todos os documentos do ChromaDB
    const q = await collection.get({
      include: ["documents"],
    });
    console.error("documents ");

    const docs = q.documents || [];


    const context = docs.join("\n\n---\n\n"); // junta todos documentos em um contexto só

    const hasContext = docs.length > 0;
    const prompt = 
      (hasContext
        ? `:\n\n${text}\n\n`
        : `.\n\n`) +
      ` ${context}\n\n` +
      ` `;

    const response = await axios.post(`${OLLAMA_HOST}/api/generate`, {
      model: "llama3.2",
      prompt,
      stream: false,
    });
    console.error("prompt response ");

    const rawText = typeof response?.data?.response === "string" ? response.data.response : "";
    const answer = rawText
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .slice(0, 4)
      .join("\n");
    console.error("prompt answer ",answer);

    res.send(answer);
  } catch (err) {
    console.error("Erro ao chamar Ollama:", err.message);
    res.status(500).send("Erro ao gerar resposta da IA.");
  }
});




app.listen(3000, () => {
  console.log("Servidor rodando na porta 3000 🚀");
});
