import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { makeChain } from '@/utils/makechain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  console.log('question', question);

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    /* create vectorstore*/
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings({}),
      {
        pineconeIndex: index,
        textKey: 'text',
        namespace: PINECONE_NAME_SPACE, //namespace comes from your config folder
      },
    );

    //create chain
    const chain = makeChain(vectorStore);
    //Ask a question using chat history

    console.log('question asked before sanitization:', question);
    console.log('question asked after sanitization:', sanitizedQuestion);

    const additionalPrompt =
      'Once find out the data, have them in JSON format.' +
      'Desired format: Response:<your response> Data:<JSON data you summarized>';

    const response = await chain.call({
      question: sanitizedQuestion + additionalPrompt,
      chat_history: history || [],
    });

    console.log('response received ', response);
    console.log('text part of response  received:', response.text);

    //now try to do pasing into json format

    let response_position = response.text.indexOf('Response:');
    let data_position = response.text.indexOf('Data:');
    const newResponse = response.text.substring(
      response_position + 10,
      data_position != -1 ? data_position : response.text.length - 1,
    );

    // save chat data into database
    const chatDataForm = JSON.parse(response.text.substring(data_position + 6));

    console.log('chatDataForm, data in JSON format', chatDataForm);

    //next  I may want to return chatDataForm instead of the full response
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}
