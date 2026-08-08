import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
});

async function obtenerRespuesta() {
    try {
        const chat = await openai.chat.completions.create({
            model: "deepseek/deepseek-r1:free",
            messages:[{ role: "user", content: "¿Qué es el número Áureo?"}],
        });

        console.log(chat.choices[0].message.content);
    } catch(error) {
        console.error("Error en la solicitud:", error);
    }
}

obtenerRespuesta();
