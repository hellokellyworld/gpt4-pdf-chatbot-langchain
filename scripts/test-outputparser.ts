import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';
import { StructuredOutputParser } from 'langchain/output_parsers';

import { z } from 'zod';

export const run = async () => {
  try {
    // With a `StructuredOutputParser` we can define a schema for the output.

    // const parser = StructuredOutputParser.fromNamesAndDescriptions({
    //   answer: "answer to the user's question",
    //   source: "source used to answer the user's question, should be a website.",
    // });

    // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const parser = StructuredOutputParser.fromZodSchema(
      //we will have an JSON objec tfor answer being a string type
      //and sources being a string array type
      z.object({
        answer: z.string().describe("answer to the user's question"),
        sources: z
          .array(z.string())
          .describe('sources used to answer the question, should be websites.'),
      }),
    );

    const formatInstructions = parser.getFormatInstructions();

    console.log('here is the format isntructions:', formatInstructions);

    const prompt = new PromptTemplate({
      template:
        'Answer the users question as best as possible.\n{format_instructions}\n{question}',
      inputVariables: ['question'],
      partialVariables: { format_instructions: formatInstructions },
    });

    const model = new OpenAI({ temperature: 0 });

    const input = await prompt.format({
      question: 'What is the capital of France?',
    });
    const response = await model.call(input); //this direct call to OpenAI after prompt is created
    //a chain can do the creation of the prompt + call to openAI together

    console.log('Here is the input/final template:', input);

    console.log('Here is the response:', response);
    /*  //without array of strings for source
        {"answer": "Paris", "source": "https://en.wikipedia.org/wiki/Paris"}
    */

    /*   //with array of strings for sources
      { answer: 'Paris', sources: [ 'https://en.wikipedia.org/wiki/Paris' ] }
    */

    console.log('Here is the parsed outputs:', await parser.parse(response));
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to test the parser');
  }
};

(async () => {
  await run();
  console.log('parser worked!');
})();
