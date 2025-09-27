import express from "express";
import axios from "axios";
import { ChromaClient } from "chromadb";
import multer from "multer";
import fs from "fs";
import ollama from 'ollama';

 
 
// modelo de embedding
const EMB_MODEL = "all-minilm";
const OLLAMA_HOST = 'http://localhost:11434'; // ajuste se necessário
const COLLECTION_NAME = 'minha_colecao';

async function getEmbedding(text) {
  const res = await ollama.embeddings({
    model: EMB_MODEL,
    prompt: text,
  });
  return res.embedding;
}
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
  metadata: { "hnsw:space": "cosine" },
  embeddingFunction: null, // vamos gerar manualmente

  });
  console.log("Coleção 'conversas' pronta!");
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

    
  
  const resp = ollama.embeddings(model=EMB_MODEL, prompt=text)
  return resp["embedding"]
  
}

// rota principal que chama o Ollama
app.post("/ia", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).send("Texto é obrigatório");

  try {
    const response = await axios.post("http://127.0.0.1:11434/api/generate", {
      model: "llama3.2",
      prompt: text,
      stream: false,
    });

    const respData = response.data.response.toString();

    // gera embedding real da pergunta+resposta
    const embedding = await generateEmbedding(`Pergunta: ${text} | Resposta: ${respData}`);

    const docId = Date.now().toString();
    await collection.add({
      ids: [docId],
      documents: [`Pergunta: ${text} | Resposta: ${respData}`],
      embeddings: [embedding],
    });

    res.send(respData);
  } catch (err) {
    console.error("Erro ao chamar Ollama:", err.message);
    res.status(500).send("Erro ao gerar resposta da IA.");
  }
});


app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).send("Arquivo é obrigatório");

  try {
    const content = fs.readFileSync(req.file.path, "utf-8");
    const lines = content.split("\n");

    let inseridos = 0;

    for (let i = 0; i < lines.length; i++) {
      const parsed = parseLine(lines[i].trim());
      if (!parsed) continue;

      try {
        // gera embedding da linha
        // const emb = await generateEmbedding(parsed.message);
        const emb = await getEmbedding(parsed.message);

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
  const results = await collection.get();
  res.json(results);
});

// rota para buscar
app.post("/search", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).send("Query é obrigatória");

  try {
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

app.listen(3000, () => {
  console.log("Servidor rodando na porta 3000 🚀");
});
