import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), 'public', 'data', 'ml_results.json');
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    return NextResponse.json({
      models: data.models,
      model_comparison: data.model_comparison,
      top_features: data.top_features,
      ml_dataset: data.ml_dataset
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load model data' }, { status: 500 });
  }
}
