import { OpenAI } from 'langchain/llms/openai';
// import { PromptTemplate } from 'langchain/prompts';
// import { StructuredOutputParser } from 'langchain/output_parsers';

import { z } from 'zod';
// import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PromptTemplate } from 'langchain/prompts';
import { LLMChain } from 'langchain/chains';
import {
  StructuredOutputParser,
  OutputFixingParser,
} from 'langchain/output_parsers';

// import { ConversationalRetrievalQAChain } from 'langchain/chains';

// import { z } from 'zod';

//now try with an LLM chain like we had in the back-end

export const run = async () => {
  try {
    ///HERE NEW EXAMPLE https://js.langchain.com/docs/modules/model_io/output_parsers/how_to/use_with_llm_chain

    // // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const outputParser = StructuredOutputParser.fromZodSchema(
      //we will have an JSON objec tfor answer being a string type
      //and sources being a string array type
      z.object({
        answer: z.string().describe("answer to the user's question"),
        sources: z
          .array(z.string())
          .describe('sources used to answer the question, should be websites.'),
      }),
    );

    const chatModel = new OpenAI({
      modelName: 'gpt-3.5-turbo', //'gpt-4', // Or gpt-3.5-turbo
      temperature: 0, // For best results with the output fixing parser
    });

    const outputFixingParser: any = OutputFixingParser.fromLLM(
      chatModel,
      outputParser as any,
    );

    // Don't forget to include formatting instructions in the prompt!
    const prompt = new PromptTemplate({
      template: `Answer the user's question as best you can:\n{format_instructions}\n{query}`,
      inputVariables: ['query'],
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

    const answerFormattingChain = new LLMChain({
      llm: chatModel,
      prompt,
      outputKey: 'records', // For readability - otherwise the chain output will default to a property named "text"
      outputParser: outputFixingParser,
      //the chain should call the outputParser.parse() before returning result
      //as parser was given to the chain already
      verbose: true, //make it show us details of the prompt made by the chain
    });

    const result = await answerFormattingChain.call(
      //here we parse in a question object, and the key is "query" ,
      //and the PromptTemplte will use it to create the final prompt
      {
        query: 'What is the capital of France?', //NOTE here the question starts with query instead of "question"
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
