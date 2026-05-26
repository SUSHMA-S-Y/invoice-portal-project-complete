import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const mlPath = path.join(process.cwd(), 'public', 'data', 'ml_results.json');
    const mlData = JSON.parse(fs.readFileSync(mlPath, 'utf-8'));

    const lookupPath = path.join(process.cwd(), 'public', 'data', 'invoice_lookup.json');
    const lookupData = JSON.parse(fs.readFileSync(lookupPath, 'utf-8'));

    // Build lightweight dropdown items: id, vendor, date, amount
    const allIds = mlData.all_invoice_ids || [];
    const dropdownItems = allIds.map((id: string) => {
      const inv = lookupData[id];
      return {
        invoice_id: id,
        vendor_name: inv?.vendor_name || '',
        invoice_date: inv?.invoice_date || '',
        submitted_amount: inv?.submitted_amount || 0,
        is_fraud: inv?.is_fraud || false,
      };
    });

    return NextResponse.json({
      all_invoice_ids: allIds,
      dropdown_items: dropdownItems,
      demo_invoices: mlData.demo_invoices || []
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load invoices' }, { status: 500 });
  }
}
