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
  const resp = await axios.post("http://127.0.0.1:11434/api/embed", {
    model: "all-minilm",
    input: text,
  });

  // alguns endpoints retornam `embeddings: [[...]]`
  if (resp.data.embeddings) return resp.data.embeddings[0];
  throw new Error("Formato inesperado da resposta de embedding");
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

// rota para upload de arquivo .txt (chat WhatsApp)
app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).send("Arquivo é obrigatório");

  try {
    let content = fs.readFileSync(req.file.path, "utf-8");
    let lines = content.split("\n");

    let ids = [];
    let docs = [];
    let embeddings = [];
    let metadatas = [];

    for (let i = 0; i < lines.length; i++) {
      let parsed = parseLine(lines[i].trim());
      if (!parsed) continue;

      try {
        let emb = await generateEmbedding(parsed.message);

        ids.push(Date.now().toString() + "_" + i);
        docs.push(parsed.message);
        console.debug("emb:",emb.length, emb[0]);

        embeddings.push(emb);
        metadatas.push({
          phone: parsed.phone,
          datetime: parsed.datetime,
          raw: parsed.raw,
          line_index: i,
        });
      } catch (err) {
        console.error("Erro ao gerar embedding da linha:", i, err.message);
      }
    }

    console.debug("embedding da s:",embeddings.length);


    if (ids.length > 0) {
      await collection.add({
        ids,
        documents: docs,
        embeddings:embeddings,
        metadatas,
      });
    }

    res.json({ ok: true, inseridos: ids.length });
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
