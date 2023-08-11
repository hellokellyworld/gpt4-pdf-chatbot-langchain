import { OpenAI } from 'langchain/llms/openai';
import { z } from 'zod';
import { PromptTemplate } from 'langchain/prompts';
import {
  StructuredOutputParser,
  OutputFixingParser,
} from 'langchain/output_parsers';

import { ConversationalRetrievalQAWithOutputParserChain } from './conversational-retrieval-qa-chain-with-outputparser'; //'langchain/chains';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';

import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

//now try with an LLM ConversationalRetrievalQAChain like we had in the back-end
export const run = async () => {
  try {
    ///HERE NEW EXAMPLE https://js.langchain.com/docs/modules/model_io/output_parsers/how_to/use_with_llm_chain

    // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const outputParser: any = StructuredOutputParser.fromZodSchema(
      //we will have a JSON object for answer being a string type
      //and sources being a string array type
      z.object({
        carrier: z.string().describe('name of shipment carrier'),
        invoicenumber: z.string().describe('invoice number of the document'),
        TQ: z.string().describe('total M/T'),
        ETD: z.string().describe('ETD of the shipment'),
        ETA: z.string().describe('ETA of the shipment'),
        POL: z.string().describe('port of loading or POL of the shipment'),
      }),
    );

    const chatModel = new OpenAI({
      modelName: 'gpt-3.5-turbo', //'gpt-4', // Or gpt-3.5-turbo
      temperature: 0, // For best results with the output fixing parser
    });

    const outputFixingParser = OutputFixingParser.fromLLM(
      chatModel,
      outputParser,
    );

    console.log(
      'this is the format instructions:',
      outputFixingParser.getFormatInstructions(),
    );
    console.log(
      'This is my outputParser object, it should have a parse() function:',
      outputFixingParser,
    );

    //pinecone vectorstore to do the retreival from a DB

    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const vectorStore: any = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings({}),
      {
        pineconeIndex: index,
        textKey: 'text',
        namespace: PINECONE_NAME_SPACE, //namespace comes from your config folder
      },
    );

    const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:`;

    const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

\nContext: \n{context}

\nQuestion: \n{question}

\nAlso use the format instructions provided here
\nFormat Instructions: \n{format_instructions}      

\nHelpful answer:`;

    // Don't forget to include formatting instructions in the prompt!

    const COMPLEX_QA_PROMPT_TEMPLATE = new PromptTemplate({
      template: QA_PROMPT,
      inputVariables: ['question', 'context'],
      partialVariables: {
        format_instructions: outputFixingParser.getFormatInstructions(),
      },
      outputParser: outputFixingParser,
    });

    // `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    // If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    // If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    // {context}

    // Question: {question}
    // Helpful answer in markdown:`;

    const answerFormattingChain =
      //  ConversationalRetrievalQAChain
      ConversationalRetrievalQAWithOutputParserChain.fromLLM(
        chatModel,
        vectorStore.asRetriever(),
        {
          outputKey: 'records',
          // qaTemplate: QA_PROMPT,
          qaChainOptions: {
            type: 'stuff',
            prompt: COMPLEX_QA_PROMPT_TEMPLATE, //more complex tempate that uses
            // inputVariables: string[];
            // outputParser?: BaseOutputParser;
            // partialVariables?: PartialValues;
            // PromptTemplate.fromTemplate(QA_PROMPT)  //Simple template
          },
          // questionGeneratorTemplate: CONDENSE_PROMPT,
          questionGeneratorChainOptions: {
            //used for rephrasing the question based on chat history
            llm: chatModel,
            template: CONDENSE_PROMPT,
          },
          verbose: true,
          outputParser: outputParser,
          returnSourceDocuments: true, //The number of source documents returned is 4 by default
        },
      );

    const question =
      `What is the invoice number,\n` +
      `What is the carrier,\n  ` +
      `and what is the invoice number,\n ` +
      'what is total M/T.\n' +
      'what is ETD,\n' +
      'what is ETA,\n' +
      'what is POL or port of loading,\n';

    const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

    const result = await answerFormattingChain.call({
      question: sanitizedQuestion,
      chat_history: [], // history || [],
    }); //this should already have the answer formatted by the output parser
    console.log('--------------------------------\n');
    console.log('here is the parsed result:', result.text);
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to test the parser');
  }
};

(async () => {
  await run();
  console.log('parser worked!');
})();
