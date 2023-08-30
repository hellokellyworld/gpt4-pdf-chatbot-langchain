import { OpenAI } from 'langchain/llms/openai';
import { z } from 'zod';
import { PromptTemplate } from 'langchain/prompts';
import {
  StructuredOutputParser,
  OutputFixingParser,
} from 'langchain/output_parsers';

import { ConversationalRetrievalQAChain } from 'langchain/chains';

import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
// import { makeChain } from '@/utils/makechain';

import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

//now try with an LLM ConversationalRetrievalQAChain like we had in the back-end

export const run = async () => {
  try {
    ///HERE NEW EXAMPLE https://js.langchain.com/docs/modules/model_io/output_parsers/how_to/use_with_llm_chain

    // // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const outputParser: any = StructuredOutputParser.fromZodSchema(
      //we will have an JSON objec tfor answer being a string type
      //and sources being a string array type
      z.object({
        carrier: z.string().describe('name of shipment carrier'),
        invoicenumber: z.string().describe('invoice number of the document'),
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

    // Don't forget to include formatting instructions in the prompt!
    const prompt = new PromptTemplate({
      template: `Answer the user's question as best you can:\n{format_instructions}\n{question}`,
      inputVariables: ['question'],
      partialVariables: {
        format_instructions: outputFixingParser.getFormatInstructions(),
      },
    });

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

    //OLD working chain no retrieval from pinecone

    // const answerFormattingChain = new LLMChain({
    //   llm: chatModel,
    //   prompt,
    //   outputKey: 'records', // For readability - otherwise the chain output will default to a property named "text"
    //   outputParser: outputFixingParser, //the chain should call the outputParser.parse() before returning result
    //   //as parser was given to the chain already
    //   verbose: true, //make it show us details of the prompt made by the chain
    // });

    //NEW conversational retrieval chain

    //     const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    // Chat History:
    // {chat_history}
    // Follow Up Input: {question}
    // Standalone question:`;

    //     const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    // If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    // If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    // {context}

    // Question: {question}
    // Helpful answer in markdown:`;

    // Don't forget to include formatting instructions in the prompt!
    const newPrompt = new PromptTemplate({
      template: `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Also use the format instructions provided here.
\n{context}
\nQuestion: \n{question}
\nFormat Instructions: \n{format_instructions}      
`,
      inputVariables: ['question', 'context'],
      partialVariables: {
        format_instructions: outputFixingParser.getFormatInstructions(),
      },
      outputParser: outputFixingParser,
    });

    const answerFormattingChain = ConversationalRetrievalQAChain.fromLLM(
      chatModel,
      vectorStore.asRetriever(),
      {
        outputKey: 'records',
        // qaTemplate: newPrompt,
        qaChainOptions: {
          prompt: newPrompt,
        },
        // questionGeneratorChainOptions: { llm: chatModel, template: newPrompt },
        returnSourceDocuments: true, //The number of source documents returned is 4 by default
        verbose: true,
      },
    );
    const result = await answerFormattingChain.call(
      //here we parse in a question object, and the key is "query" ,
      //and the PromptTemplte will use it to create the final prompt
      {
        question: 'What is the invoice number, and what is the carrier',
      },
    );

    // /*   //with array of strings for sources
    //   { answer: 'Paris', sources: [ 'https://en.wikipedia.org/wiki/Paris' ] }
    // */

    console.log(JSON.stringify(result.records, null, 2)); //this should already have the answer formatted by the output parser

    // console.log('Here is the parsed outputs:', await parser.parse(response));
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to test the parser');
  }
};

(async () => {
  await run();
  console.log('parser worked!');
})();
