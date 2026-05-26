import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(request: Request) {
  try {
    const { invoiceId } = await request.json();

    const lookupPath = path.join(process.cwd(), 'public', 'data', 'invoice_lookup.json');
    const lookupData = JSON.parse(fs.readFileSync(lookupPath, 'utf-8'));

    const invoice = lookupData[invoiceId] || null;

    if (!invoice) {
      return NextResponse.json({ error: 'Invoice not found', invoice_id: invoiceId }, { status: 404 });
    }

    // Ensure ALL required fields exist with safe defaults
    const safeInvoice = {
      invoice_id: invoice.invoice_id || invoiceId || 'UNKNOWN',
      vendor_name: invoice.vendor_name || '',
      vendor_id: invoice.vendor_id || '',
      invoice_date: invoice.invoice_date || '',
      submitted_amount: typeof invoice.submitted_amount === 'number' ? invoice.submitted_amount : 0,
      approved_amount: typeof invoice.approved_amount === 'number' ? invoice.approved_amount : 0,
      bank_account: invoice.bank_account || '',
      is_fraud: !!invoice.is_fraud,
      risk_score: typeof invoice.risk_score === 'number' ? invoice.risk_score : 0,
      findings: typeof invoice.findings === 'number' ? invoice.findings : 0,
      exposure: typeof invoice.exposure === 'number' ? invoice.exposure : 0,
      status: invoice.status || 'CLEARED',
      anomaly_type: invoice.anomaly_type || null,
      anomaly_detail: invoice.anomaly_detail || null,
      risk_level: invoice.risk_level || null,
      tags: invoice.tags || null,
      ref: invoice.ref || '#AUD-2025-0000',
      ensemble_probability: typeof invoice.ensemble_probability === 'number' ? invoice.ensemble_probability : 0,
      model_votes: invoice.model_votes || {},
      features: invoice.features || {
        invoice_amount: null, tax_amount: null, processing_days: null,
        approval_level: null, vendor_reliability: null, deviation_from_avg: null, amount_zscore: null,
      }
    };

    return NextResponse.json({
      ...safeInvoice,
      generated_date: new Date().toISOString().split('T')[0],
      analysis_timestamp: new Date().toISOString()
    });
  } catch (error) {
    return NextResponse.json({ error: 'Verification failed' }, { status: 500 });
  }
}
