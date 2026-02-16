COLUMN_MAPPING_PROMPT = """ Role: You are a Data Integration Expert specializing in pharmaceutical supply chain systems. Task: Analyze a list of column headers from a medicine distributor's catalog and map them to our standardized schema.

Input Headers: {headers}

Mapping Logic:

raw_name: Identify the column containing the product name or description.

base_price: Identify the column for the gross/unit price (before discounts).

discount: Identify the column representing percentage or nominal discounts. If not explicitly found, return null.

confidence_score: Assign a value between 0.0 and 1.0 based on how certain you are of this mapping.

Output Requirement: Return ONLY a valid JSON object. Do not include introductory text, markdown code blocks, or explanations.

Target Schema: {{ "raw_name": "string", "base_price": "string", "discount": "string or null", "confidence_score": float }} """
