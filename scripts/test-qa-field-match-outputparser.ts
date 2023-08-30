//test the ConversationalRetrievalQAWithOutputParserChain along with
//field matching propmpts for question and answer
import { OpenAI } from 'langchain/llms/openai';
import { z } from 'zod';
import { PromptTemplate } from 'langchain/prompts';
import { zodToJsonSchema } from 'zod-to-json-schema';
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
    const dataFields = ['Invoice_Number', 'Quantity'];
    const dataFieldsDescription = [
      'a unique, sequential code that is systematically assigned to invoices',
      'total M/T',
    ];
    var extractAllResult: any = {};
    var resultsArray: any[] = [];
    var obj: any = {};
    dataFields.map((item, index) => {
      obj[item] = z.string().describe(dataFieldsDescription[index]).optional();
    });

    const schema = z.object(obj);

    //define output praser from schema
    const outputParser: any = StructuredOutputParser.fromZodSchema(schema);

    //get the JSON format of the schema

    //set the required part to [] for the JSON format of the schema

    //use the schema to get format instructions with a place holder for the required:[] entry

    //create the formatting instruction for the schema

    //create the the model

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

    //get the vector store

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

    const QUESTION_GENERATION_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

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

    //here parial variables are used to intialize a prompt with varables
    //then the final call just need to use the rest of the variables that are defined
    //in inputVariables
    //so that the template QA_PROMPT has many variables in it, and the format_instructions are
    //already initialized here so that the finall call does not have to have
    //the format_instructions passed in any more

    //inthis case we should NOT need the format_instructions as partial variables here
    //instead we should use regular expression to earch and replace it later

    const COMPLEX_QA_PROMPT_TEMPLATE = new PromptTemplate({
      template: QA_PROMPT,
      inputVariables: ['question', 'context', 'format_instructions'],
      // partialVariables: {
      //   format_instructions: format_instructions, //outputFixingParser.getFormatInstructions(),
      // },
      outputParser: outputFixingParser,
    });

    //get a question schema field matching prompt

    const SCHEMA_MATCHING_PROMPT = `You are a helpful AI assisnant that can match JSON data schema fields with a question. Here is a json schema, please let me know which field in the following schema matches the best with the questions asked. 
  \n\n JSON data schema:\n
  \`\`\` \n\n
  {schematext}
  \n\n
  \`\`\`
  \n\nPlease provide the accurate and exact field name as answer. Also use the format instructions provided here for the answer. 
  \n\n Format Instructions: \n\nYou must format your output as a JSON value that adheres to a given \"JSON schema\" instance.\"JSON Schema\" is a declarative language that allows you to annotate and validate JSON documents.\n\nFor example, the example \"JSON Schema\" instance {{\"properties\": {{\"foo\": {{\"description\": \"a list of test words\", \"type\": \"array\", \"items\": {{\"type\": \"string\"}}}}}}, \"required\": [\"foo\"]}}}}\nwould match an object with one required property, \"foo\". The \"type\" property specifies \"foo\" must be an \"array\", and the \"description\" property semantically describes it as \"a list of test words\". The items within \"foo\" must be strings.\nThus, the object {{\"foo\": [\"bar\", \"baz\"]}} is a well-formatted instance of this example \"JSON Schema\". 
  The object {{\"properties\": {{\"foo\": [\"bar\", \"baz\"]}}}} is not well-formatted.\n\nYour output will be parsed and type-checked according to the provided schema instance, so make sure all fields in your output match the schema exactly and there are no trailing commas!
  
  \n\nHere is the JSON Schema instance your output must adhere to. Include the enclosing markdown codeblock:\n
  \`\`\` \n\n
  {outputschematext}
  \n\n
  \`\`\` 
 
  \nQuestion: {question}
  \n   
  \nAnswer:`;

    //get a chain that can be used to get the schema field

    console.log('Tom Long: before getting prompt');
    const prompt = new PromptTemplate({
      template: SCHEMA_MATCHING_PROMPT,
      inputVariables: ['question', 'schematext', 'outputschematext'],
    });

    console.log('--------------------------------\n');
    console.log('here is the prompt\n', prompt);
    console.log('--------------------------------\n');

    let schemaObj: any = zodToJsonSchema(schema);

    console.log('--------------------------------\n');
    console.log('Long14:here is the schema after zod: \n', schemaObj);
    console.log('--------------------------------\n');

    const requiredFields: string[] = schemaObj.required
      ? schemaObj.required
      : [];
    //set the required field to empty
    schemaObj.required = [];

    console.log('--------------------------------\n');
    console.log(
      'Long:15 here is the schema after setting required to empty: \n',
      schemaObj,
    );
    console.log('--------------------------------\n');

    const stringifiedSchema = JSON.stringify(schemaObj);

    console.log('--------------------------------\n');
    console.log('here is the stringified schema text \n', stringifiedSchema);
    console.log('--------------------------------\n');

    //next lets try to format the prompt and see if there is a final one

    const schematext = 'json\n' + stringifiedSchema + '\n';

    const outputschematext =
      // '{"field_name":"name of field that matches the question"}';
      `json\n{\"field_name\":"name of the field"}\n`;

    //get the chain that has the template ready

    const qaFMChain =
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
            template: QUESTION_GENERATION_PROMPT,
          },
          verbose: true,
          outputParser: outputParser,
          returnSourceDocuments: true, //The number of source documents returned is 4 by default
        },
      );

    //get a Simple question
    // const question = `What is the invoice number,\n`; //+
    // `What is the carrier,\n  ` +
    // `and what is the invoice number,\n ` +
    // 'what is total M/T.\n' +
    // 'what is ETD,\n' +
    // 'what is ETA,\n' +
    // 'what is POL or port of loading,\n';
    console.log('extractAllResult');
    console.log('extractAllResult');
    console.log('extractAllResult');
    console.log('extractAllResult');
    console.log('extractAllResult');
    console.log('extractAllResult');

    await Promise.all(
      dataFields.map(async (field) => {
        const question = `what is ${field},\n`;
        const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

        //use the simple question to find the matching field in the schema

        const finalPrompt = await prompt.format({
          question: sanitizedQuestion,
          schematext: schematext,
          outputschematext: outputschematext,
        });

        console.log('--------------------------------\n');
        console.log('here is the final prompt\n', finalPrompt);
        console.log('--------------------------------\n');

        //next we test usage of the final prompt by calling the model
        const response = await chatModel.generate([finalPrompt]);
        console.log('--------------------------------\n');
        console.log(
          'here is the final answer\n',
          response.generations[0][0].text,
        );
        console.log('--------------------------------\n');

        //retrieve the filed name in a JSON object
        const fieldNameObj: {
          field_name: string;
        } = JSON.parse(response.generations[0][0].text);

        const fieldName: string = fieldNameObj.field_name;

        console.log('--------------------------------\n');
        console.log('here is the final field name value:\n', fieldName);
        console.log('--------------------------------\n');

        //now we want to get the final prompt by using the fieldName found

        const format_instructions =
          getFormatInstructionsWithoutRequiredField(schema);

        //create the QA prompt template with the place holder for
        //question, context, format_instructions and the format instrucitons should
        //have the required field to be filled out later

        console.log(
          'Tom Long13: here is the instructions with place holder' +
            format_instructions,
        );

        //now we use regular expression to replace the value of the place holder

        //this format_instruction has a place holder, now lets use regular expression
        //to fill out the place holder instead

        const final_format_instructins = format_instructions.replaceAll(
          'PLACE_HOLDER_FOR_EQUIRED_FIELDS',
          fieldName,
        );

        // ('PLACE_HOLDER_FOR_EQUIRED_FIELDS', fieldName);
        // fieldName);

        //add the required field as a template variable in addition to the
        //question, context and chat_history
        //note that the context is filled in during the call,
        //hence the required_field can be filled in during the call as well.

        const finalQAPrompt = await COMPLEX_QA_PROMPT_TEMPLATE.format({
          question: sanitizedQuestion,
          context: 'EMPTY CONTEXT',
          chat_history: [],
          format_instructions: final_format_instructins,
        });

        console.log('--------------------------------\n');
        console.log('Long12: here is the final QA prompt:', finalQAPrompt);

        //then we go call the langchain to get the results
        const result = await qaFMChain.call({
          question: sanitizedQuestion,
          chat_history: [], // history || [],
          format_instructions: final_format_instructins,
        }); //this should already have the answer formatted by the output parser
        console.log('--------------------------------\n');
        console.log('here is the parsed result:', result.text);
        const obj1 = result.text;
        for (var key in obj1) {
          // console.log(key);
          // console.log(obj1[key]);
          extractAllResult[key] = obj1[key];
        }
      }),
    );
    console.log(extractAllResult);
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to test the parser');
  } finally {
  }
};

const getFormatInstructionsWithoutRequiredField = (schema: any) => {
  let schemaJSONObject: any = zodToJsonSchema(schema); //convert the schema to a JSON object

  //set the required part to empty first

  const mystr = '{"required":"[{required_field}]"}';
  const strJSON = JSON.parse(mystr);

  console.log('--------------------------------\n');
  console.log('Long10:here is the strJSON:\n', strJSON);

  schemaJSONObject.required = ['PLACE_HOLDER_FOR_EQUIRED_FIELDS'];
  // strJSON; // ['`{ required_field }`']; //this required field will be filled in later

  console.log(
    'Tom Long11: Here is is the schemaJSONObject:\n',
    JSON.stringify(schemaJSONObject),
  );

  return `You must format your output as a JSON value that adheres to a given "JSON Schema" instance.

"JSON Schema" is a declarative language that allows you to annotate and validate JSON documents.

For example, the example "JSON Schema" instance {{"properties": {{"foo": {{"description": "a list of test words", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
would match an object with one required property, "foo". The "type" property specifies "foo" must be an "array", and the "description" property semantically describes it as "a list of test words". The items within "foo" must be strings.
Thus, the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of this example "JSON Schema". The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Your output will be parsed and type-checked according to the provided schema instance, so make sure all fields in your output match the schema exactly and there are no trailing commas!

Here is the JSON Schema instance your output must adhere to. Include the enclosing markdown codeblock:
\`\`\`json
${JSON.stringify(schemaJSONObject)}
\`\`\`\n

Just provide the required info here and not all fields defined in the schema above. \"required\":[\"PLACE_HOLDER_FOR_EQUIRED_FIELDS\"]

`;
};

(async () => {
  await run();
  console.log('parser worked!');
})();
