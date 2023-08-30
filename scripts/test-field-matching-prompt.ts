import { OpenAI } from 'langchain/llms/openai';
import { z as zod } from 'zod';
import { PromptTemplate } from 'langchain/prompts';
import { zodToJsonSchema } from 'zod-to-json-schema';

import { LLMChain, LLMChainInput } from 'langchain/chains';

//now try with an LLM ConversationalRetrievalQAChain like we had in the back-end
export const run = async () => {
  try {
    ///HERE NEW EXAMPLE https://js.langchain.com/docs/modules/model_io/output_parsers/how_to/use_with_llm_chain

    // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
    const schema = zod.object({
      commodity_name: zod.string().describe('name of commodity'),
      shipment_carrier: zod.string().describe('name of shipment carrier'),
      shipment_vessel: zod.string().describe('name of shipment vessel'),
      shipment_voyage: zod.string().describe('name of shipment voyage'),
      cargo_value: zod.number().describe('total value in USD'),
      cargo_quantity: zod
        .number()
        .describe('total quantity in pieces or bales or totes etc'),
      cargo_weight: zod.number().describe('total weight in MT'),
      cargo_number_of_containers: zod.number().describe('number of containers'),
      beneficiary: zod.string().describe('seller name'),
      seller_name: zod.string().describe('seller name'),
      consignee_name: zod.string().describe('buyer name or consignee name'),
      port_of_discharge: zod.string().describe('port of discharge'),
      port_of_loading: zod.string().describe('port of loading'),
      invoice_number: zod.string().describe('invoice number'),
      ETD: zod.string().describe('ETD of the shipment'),
      ETA: zod.string().describe('ETA of the shipment'),
      insured_value: zod.number().describe('insured value'),
    });

    const chatModel = new OpenAI({
      modelName: 'gpt-3.5-turbo', //'gpt-4', // Or gpt-3.5-turbo
      temperature: 0, // For best results with the output fixing parser
    });

    //new chain without parser

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
    console.log('here is the schema after zod: \n', schemaObj);
    console.log('--------------------------------\n');

    const requiredFields: string[] = schemaObj.required;
    //set the required field to empty
    schemaObj.required = [];

    console.log('--------------------------------\n');
    console.log(
      'here is the schema after setting required to empty: \n',
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

    const question = 'extract invoice number';

    const finalPrompt = await prompt.format({
      question: question,
      schematext: schematext,
      outputschematext: outputschematext,
    });

    console.log('--------------------------------\n');
    console.log('here is the final prompt\n', finalPrompt);
    console.log('--------------------------------\n');

    //next we test usage of the final prompt by calling the model
    const response = await chatModel.generate([finalPrompt]);
    console.log('--------------------------------\n');
    console.log('here is the final answer\n', response.generations[0][0].text);
    console.log('--------------------------------\n');

    //retrieve the filed name in a JSON object
    const filedNameObj: { field_name: string } = JSON.parse(
      response.generations[0][0].text,
    );

    const filedName: string = filedNameObj.field_name;

    console.log('--------------------------------\n');
    console.log('here is the final field name value:\n', filedName);
    console.log('--------------------------------\n');

    const fieldIsGood: boolean = requiredFields.includes(filedName);

    console.log('--------------------------------\n');
    console.log('here is fieldIsGood:\n', fieldIsGood);
    console.log('--------------------------------\n');

    if (!fieldIsGood) {
      throw new Error('Failed to get the correct field');
    }

    const fieldFindingChain = new LLMChain({ llm: chatModel, prompt: prompt });

    const newResonse = await fieldFindingChain.call({
      question: question,
      schematext: schematext,
      outputschematext: outputschematext,
    });
    console.log('--------------------------------\n');
    console.log('here is new resonse\n', newResonse);
    console.log('--------------------------------\n');

    //retrieve the filed name in a JSON object
    const newfiledNameObj: { field_name: string } = JSON.parse(newResonse.text);

    const newfiledName: string = newfiledNameObj.field_name;

    console.log('--------------------------------\n');
    console.log('here is the new field name value:\n', newfiledName);
    console.log('--------------------------------\n');

    const newfieldIsGood: boolean = requiredFields.includes(newfiledName);

    console.log('--------------------------------\n');
    console.log('here is newfieldIsGood:\n', newfieldIsGood);
    console.log('--------------------------------\n');

    if (!newfieldIsGood) {
      throw new Error('Failed to get the correct field');
    }

    //We may want to use zod records for dynamic schema
    // https://stackoverflow.com/questions/75373940/how-do-i-create-a-zod-object-with-dynamic-keys
    // https://github.com/colinhacks/zod#records

    //see if we can parse out the results here too
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to test the prompt');
  }
};

(async () => {
  await run();
  console.log('prompt is here, matching field name is here!');
})();
