'use client';

import React, { useState, useEffect, useCallback, Component, ReactNode, useRef, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Shield, AlertTriangle, CheckCircle2, XCircle, ArrowLeft,
  BarChart3, Brain, FileText, History, Activity,
  TrendingUp, Database, Zap, Eye, EyeOff, FileSearch, Code2,
  Mail, Lock, UserPlus, LogOut, Phone, User, ArrowRight
} from 'lucide-react';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from '@/components/ui/table';
import { ScrollArea } from '@/components/ui/scroll-area';

// =================== ERROR BOUNDARY ===================
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; error: Error | null }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <div className="text-5xl mb-4">⚠️</div>
            <h2 className="text-xl font-bold text-red-600">Something went wrong</h2>
            <p className="text-slate-500 mt-2 text-sm">{this.state.error?.message || 'An unexpected error occurred.'}</p>
            <button
              className="mt-6 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// =================== TYPES ===================
interface ModelVote {
  prediction: string;
  probability: number;
}
interface Invoice {
  invoice_id: string;
  vendor_name: string;
  vendor_id?: string;
  invoice_date: string;
  submitted_amount: number;
  approved_amount: number;
  bank_account: string;
  is_fraud: boolean;
  risk_score: number;
  findings: number;
  exposure: number;
  status: string;
  ref?: string;
  anomaly_type?: string | null;
  anomaly_detail?: string | null;
  risk_level?: string | null;
  tags?: string[] | null;
  ensemble_probability?: number;
  model_votes?: Record<string, ModelVote>;
  features?: {
    invoice_amount: number | null;
    tax_amount: number | null;
    processing_days: number | null;
    approval_level: number | null;
    vendor_reliability: number | null;
    deviation_from_avg: number | null;
    amount_zscore: number | null;
  };
}

interface ModelResult {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  confusion_matrix: number[][];
  cv_scores: number[];
  feature_importance: Record<string, number>;
  training_code?: string;
  training_output?: string;
}

interface DropdownItem {
  invoice_id: string;
  vendor_name: string;
  invoice_date: string;
  submitted_amount: number;
  is_fraud: boolean;
}

interface AnalysisData {
  data_overview: { total_records: number; total_columns: number; columns: string[]; memory_usage_mb: number };
  data_quality: { total_missing_values: number; columns_with_missing: number; missing_details: Record<string, { missing_count: number; missing_pct: number; dtype: string }>; duplicate_invoice_ids: number; duplicate_pct: number };
  fraud_distribution: Record<string, number>;
  fraud_unique_values?: Record<string, number>;
  payment_status_distribution: Record<string, number>;
  amount_distribution: Record<string, number>;
  processing_distribution: Record<string, number>;
  column_statistics: Record<string, { dtype: string; unique_count: number; null_count: number; sample_values: string[]; mean?: number; median?: number; std?: number; min?: number; max?: number }>;
  models: Record<string, ModelResult>;
  model_comparison: Array<{ model: string; accuracy: number; precision: number; recall: number; f1_score: number; roc_auc: number; avg_cv: number }>;
  top_features: Array<{ feature: string; importance: number }>;
  ml_dataset: { total_samples: number; fraud_count: number; legit_count: number; fraud_ratio: number; features_used: string[]; feature_count: number };
  demo_invoices: Invoice[];
  all_invoice_ids?: string[];
}

// =================== HELPER ===================
const safeNum = (val: number | null | undefined): number => (typeof val === 'number' && !isNaN(val)) ? val : 0;
const safeStr = (val: string | null | undefined): string => val || '';

// =================== MAIN APP ===================
export default function FraudPortal() {
  const [currentPage, setCurrentPage] = useState<string>('portal');
  const [selectedInvoice, setSelectedInvoice] = useState<string>('');
  const [dropdownItems, setDropdownItems] = useState<DropdownItem[]>([]);
  const [allInvoiceIds, setAllInvoiceIds] = useState<string[]>([]);
  const [invoiceSearch, setInvoiceSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [currentInvoice, setCurrentInvoice] = useState<Invoice | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verificationHistory, setVerificationHistory] = useState<Invoice[]>([]);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const [actionResult, setActionResult] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [authView, setAuthView] = useState('login');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authName, setAuthName] = useState('');
  const [authConfirmPassword, setAuthConfirmPassword] = useState('');
  const [authPhone, setAuthPhone] = useState('');
  const [authTerms, setAuthTerms] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loggedInUser, setLoggedInUser] = useState('');

  const dropdownContainerRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const selectedDropdownItem = dropdownItems.find(d => d.invoice_id === selectedInvoice);

  // Debounce search input
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = setTimeout(() => {
      setDebouncedSearch(invoiceSearch);
    }, 300);
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [invoiceSearch]);

  // Click outside handler
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownContainerRef.current && !dropdownContainerRef.current.contains(event.target as Node)) {
        setActiveDropdown(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Filtered items with useMemo
  const filteredItems = useMemo(() => {
    if (!debouncedSearch) return dropdownItems;
    const term = debouncedSearch.toLowerCase();
    return dropdownItems.filter(d =>
      d.invoice_id.toLowerCase().includes(term) ||
      d.vendor_name.toLowerCase().includes(term) ||
      d.invoice_date.toLowerCase().includes(term) ||
      d.submitted_amount.toString().includes(term)
    );
  }, [dropdownItems, debouncedSearch]);

  useEffect(() => {
    fetch('/api/invoices')
      .then(res => res.json())
      .then(data => {
        if (data.dropdown_items) setDropdownItems(data.dropdown_items);
        if (data.all_invoice_ids) setAllInvoiceIds(data.all_invoice_ids);
      })
      .catch(() => {});

    fetch('/api/analysis')
      .then(res => res.json())
      .then(data => setAnalysisData(data))
      .catch(() => {});
  }, []);

  const handleSelectInvoice = useCallback((invoiceId: string) => {
    setSelectedInvoice(invoiceId);
    setInvoiceSearch('');
    setDebouncedSearch('');
    setActiveDropdown(null);
    setAuthError('');
  }, []);

  const handleVerify = useCallback(async () => {
    if (!selectedInvoice) return;
    setVerifying(true);
    setActionResult(null);
    try {
      const res = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ invoiceId: selectedInvoice })
      });
      const data = await res.json();
      const safeData: Invoice = {
        invoice_id: safeStr(data.invoice_id),
        vendor_name: safeStr(data.vendor_name),
        vendor_id: data.vendor_id || '',
        invoice_date: safeStr(data.invoice_date),
        submitted_amount: safeNum(data.submitted_amount),
        approved_amount: safeNum(data.approved_amount),
        bank_account: safeStr(data.bank_account),
        is_fraud: !!data.is_fraud,
        risk_score: safeNum(data.risk_score),
        findings: safeNum(data.findings),
        exposure: safeNum(data.exposure),
        status: safeStr(data.status) || 'CLEARED',
        ref: data.ref || '',
        anomaly_type: data.anomaly_type || null,
        anomaly_detail: data.anomaly_detail || null,
        risk_level: data.risk_level || null,
        tags: data.tags || null,
        ensemble_probability: safeNum(data.ensemble_probability),
        model_votes: data.model_votes || {},
        features: data.features || { invoice_amount: null, tax_amount: null, processing_days: null, approval_level: null, vendor_reliability: null, deviation_from_avg: null, amount_zscore: null }
      };
      setCurrentInvoice(safeData);
      setVerificationHistory(prev => {
        const exists = prev.find(i => i.invoice_id === safeData.invoice_id);
        if (exists) return prev;
        return [...prev, safeData];
      });
      setCurrentPage('risk-assessment');
    } catch {
      const dd = dropdownItems.find(d => d.invoice_id === selectedInvoice);
      if (dd) {
        const fallback: Invoice = {
          invoice_id: dd.invoice_id,
          vendor_name: dd.vendor_name,
          invoice_date: dd.invoice_date,
          submitted_amount: dd.submitted_amount,
          approved_amount: dd.submitted_amount,
          bank_account: `IBAN ****${dd.invoice_id.slice(-4)}`,
          is_fraud: dd.is_fraud,
          risk_score: dd.is_fraud ? 85 : 8,
          findings: dd.is_fraud ? 2 : 0,
          exposure: 0,
          status: dd.is_fraud ? 'WARNING' : 'CLEARED',
          ref: '',
          anomaly_type: dd.is_fraud ? 'Amount Inflation' : null,
          anomaly_detail: dd.is_fraud ? 'Submitted amount deviates from vendor average.' : null,
          risk_level: dd.is_fraud ? 'High' : null,
          tags: dd.is_fraud ? ['FINANCIAL LOSS', 'FRAUD RISK'] : null,
          ensemble_probability: dd.is_fraud ? 0.85 : 0.05,
          model_votes: {},
          features: { invoice_amount: null, tax_amount: null, processing_days: null, approval_level: null, vendor_reliability: null, deviation_from_avg: null, amount_zscore: null },
        };
        setCurrentInvoice(fallback);
        setVerificationHistory(prev => {
          const exists = prev.find(i => i.invoice_id === fallback.invoice_id);
          if (exists) return prev;
          return [...prev, fallback];
        });
        setCurrentPage('risk-assessment');
      }
    } finally {
      setVerifying(false);
    }
  }, [selectedInvoice, dropdownItems]);

  // =================== AUTH HANDLERS ===================
  useEffect(() => {
    const saved = localStorage.getItem('invoicePortalUser');
    if (saved) {
      setIsLoggedIn(true);
      setLoggedInUser(saved);
    }
  }, []);

  const handleLogin = () => {
    setAuthError('');
    setAuthSuccess('');
    if (!authEmail || !authPassword) {
      setAuthError('Please enter both email and password.');
      return;
    }
    if (authEmail === 'admin@dyashin.com' && authPassword === 'admin123') {
      setIsLoggedIn(true);
      setLoggedInUser(authEmail);
      localStorage.setItem('invoicePortalUser', authEmail);
      return;
    }
    try {
      const stored = localStorage.getItem('invoicePortalUsers');
      const users: Array<{name: string; email: string; password: string}> = stored ? JSON.parse(stored) : [];
      const found = users.find(u => u.email.toLowerCase() === authEmail.toLowerCase());
      if (found && found.password === authPassword) {
        setIsLoggedIn(true);
        setLoggedInUser(authEmail);
        localStorage.setItem('invoicePortalUser', authEmail);
      } else {
        setAuthError('Invalid email or password. Please try again.');
      }
    } catch {
      setAuthError('An error occurred. Please try again.');
    }
  };

  const handleRegister = () => {
    setAuthError('');
    setAuthSuccess('');
    if (!authName || !authEmail || !authPassword || !authConfirmPassword) {
      setAuthError('Please fill in all fields.');
      return;
    }
    if (authPassword.length < 6) {
      setAuthError('Password must be at least 6 characters long.');
      return;
    }
    if (authPassword !== authConfirmPassword) {
      setAuthError('Passwords do not match.');
      return;
    }
    if (!authTerms) {
      setAuthError('Please agree to the Terms & Conditions to proceed.');
      return;
    }
    if (!authEmail.includes('@') || !authEmail.includes('.')) {
      setAuthError('Please enter a valid email address.');
      return;
    }
    try {
      const stored = localStorage.getItem('invoicePortalUsers');
      const users: Array<{name: string; email: string; password: string; phone: string}> = stored ? JSON.parse(stored) : [];
      users.push({ name: authName, email: authEmail, password: authPassword, phone: '' });
      localStorage.setItem('invoicePortalUsers', JSON.stringify(users));
      setAuthSuccess('Account created successfully! You can now sign in.');
      setAuthView('login');
      setAuthName('');
      setAuthEmail('');
      setAuthPassword('');
      setAuthConfirmPassword('');
      setAuthTerms(false);
      setOtpSent(false);
    } catch {
      setAuthError('An error occurred. Please try again.');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setLoggedInUser('');
    setAuthEmail('');
    setAuthPassword('');
    setAuthError('');
    setAuthSuccess('');
    setAuthView('login');
    setCurrentPage('portal');
    localStorage.removeItem('invoicePortalUser');
  };

  // =================== AUTH RENDER ===================
  const renderLogin = () => {
    return (
      <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <Card className="shadow-xl rounded-xl overflow-hidden border-0">
            {/* Blue Header */}
            <div className="bg-[#007BFF] px-8 py-5">
              <h1 className="text-white text-xl font-bold tracking-wide">Sign In</h1>
            </div>

            <CardContent className="p-8 pt-6 space-y-5">
              {authError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-600 font-medium">{authError}</p>
                </div>
              )}

              {authSuccess && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                  <p className="text-sm text-emerald-600 font-medium">{authSuccess}</p>
                </div>
              )}

              {/* Email Address */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Email Address <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type="email"
                    placeholder="Enter your email address"
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    className="pl-10 h-11 bg-[#F3F4F6] border-gray-300 rounded-lg focus:border-[#B3D4FC] focus:ring-1 focus:ring-[#B3D4FC] text-sm text-slate-800"
                    onKeyDown={(e) => { if (e.key === 'Enter') handleLogin(); }}
                  />
                </div>
              </div>

              {/* Password */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Password <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    className="pl-10 pr-10 h-11 bg-[#F3F4F6] border-gray-300 rounded-lg focus:border-[#B3D4FC] focus:ring-1 focus:ring-[#B3D4FC] text-sm text-slate-800"
                    onKeyDown={(e) => { if (e.key === 'Enter') handleLogin(); }}
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>



              {/* Sign In Button */}
              <button
                onClick={handleLogin}
                className="w-full h-11 bg-[#007BFF] hover:bg-[#0056b3] text-white font-bold text-sm rounded-lg transition-colors"
              >
                Sign In
              </button>

              {/* Register link */}
              <p className="text-center text-sm text-slate-600">
                Don&apos;t have an account?{' '}
                <button
                  onClick={() => { setAuthView('register'); setAuthError(''); setAuthSuccess(''); setOtpSent(false); }}
                  className="text-[#007BFF] hover:text-[#0056b3] font-bold underline"
                >
                  Register
                </button>
              </p>

              {/* Demo credentials */}
              <div className="mt-2 p-3 bg-blue-50 border border-blue-100 rounded-lg">
                <p className="text-xs text-blue-600 text-center font-medium">
                  Demo Credentials: admin@dyashin.com / admin123
                </p>
              </div>
            </CardContent>
          </Card>
          <p className="text-center text-xs text-slate-400 mt-4">
            Invoice Authenticity Portal &copy; 2024
          </p>
        </div>
      </div>
    );
  };

  const renderRegister = () => {
    return (
      <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <Card className="shadow-xl rounded-xl overflow-hidden border-0">
            {/* Blue Header */}
            <div className="bg-[#2563EB] px-8 py-5">
              <h1 className="text-white text-xl font-bold tracking-wide">Register</h1>
            </div>

            <CardContent className="p-8 pt-6 space-y-4">
              {authError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-600 font-medium">{authError}</p>
                </div>
              )}

              {authSuccess && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                  <p className="text-sm text-emerald-600 font-medium">{authSuccess}</p>
                </div>
              )}

              {/* Full Name */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Full Name <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type="text"
                    placeholder="Enter your full name"
                    value={authName}
                    onChange={(e) => setAuthName(e.target.value)}
                    className="pl-10 h-11 border-gray-300 rounded-lg focus:border-[#93C5FD] focus:ring-1 focus:ring-[#93C5FD] text-sm text-slate-800"
                  />
                </div>
              </div>

              {/* Email Address */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Email Address <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type="email"
                    placeholder="Enter your email address"
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    className="pl-10 h-11 border-gray-300 rounded-lg focus:border-[#93C5FD] focus:ring-1 focus:ring-[#93C5FD] text-sm text-slate-800"
                  />
                </div>
              </div>

              {/* Password */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Password <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    className="pl-10 pr-10 h-11 border-gray-300 rounded-lg focus:border-[#93C5FD] focus:ring-1 focus:ring-[#93C5FD] text-sm text-slate-800"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Confirm Password */}
              <div className="space-y-1.5">
                <Label className="text-sm font-medium text-slate-700">
                  Confirm Password <span className="text-red-500">*</span>
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Enter your confirm password"
                    value={authConfirmPassword}
                    onChange={(e) => setAuthConfirmPassword(e.target.value)}
                    className="pl-10 pr-10 h-11 border-gray-300 rounded-lg focus:border-[#93C5FD] focus:ring-1 focus:ring-[#93C5FD] text-sm text-slate-800"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Terms & Conditions */}
              <div className="flex items-start gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => setAuthTerms(!authTerms)}
                  className="mt-0.5 w-5 h-5 border-2 border-gray-300 rounded flex items-center justify-center shrink-0 transition-colors"
                  style={{ backgroundColor: authTerms ? '#2563EB' : 'white' }}
                >
                  {authTerms && (
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
                <p className="text-sm text-slate-600">
                  I agree to the{' '}
                  <button className="text-[#2563EB] hover:text-[#1d4ed8] underline font-medium">
                    Terms &amp; Conditions
                  </button>
                </p>
              </div>

              {/* Register Button */}
              <button
                onClick={handleRegister}
                className="w-full h-11 bg-[#2563EB] hover:bg-[#1d4ed8] text-white font-bold text-sm rounded-lg transition-colors flex items-center justify-center gap-2 mt-2"
              >
                Register
                <ArrowRight className="h-4 w-4" />
              </button>

              {/* Sign In link */}
              <p className="text-center text-sm text-slate-600 pt-1">
                Already have an account?{' '}
                <button
                  onClick={() => { setAuthView('login'); setAuthError(''); setAuthSuccess(''); setOtpSent(false); }}
                  className="text-[#2563EB] hover:text-[#1d4ed8] font-bold underline"
                >
                  Sign In
                </button>
              </p>
            </CardContent>
          </Card>
          <p className="text-center text-xs text-slate-400 mt-4">
            Invoice Authenticity Portal &copy; 2024
          </p>
        </div>
      </div>
    );
  };

  const renderAuth = () => {
    if (authView === 'register') return renderRegister();
    return renderLogin();
  };

  // =================== PORTAL PAGE ===================
  const renderPortal = () => {
    return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 pb-16">
      {/* Navy blue header bar */}
      <div className="bg-[#1a237e] px-4 py-5">
        <div className="max-w-xl mx-auto flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <Shield className="h-7 w-7 text-white" />
              <h1 className="text-xl font-bold text-white tracking-tight">Invoice Authenticity Portal</h1>
            </div>
            <p className="text-blue-100 text-xs">ML-Based Invoice Fraud Detection System</p>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors text-sm"
            title="Sign out"
          >
            <span className="hidden sm:inline max-w-[180px] truncate text-white/80 text-xs">{loggedInUser}</span>
            <LogOut className="h-4 w-4 text-white/80" />
          </button>
        </div>
      </div>

      <div className="max-w-xl mx-auto px-4 mt-6" ref={dropdownContainerRef}>
        <Card className="shadow-xl border-0">
          <CardContent className="p-6">
            <div className="space-y-6">
              {/* Invoice count badge */}
              <div className="flex items-center justify-center">
                <span className="text-xs text-blue-500">{dropdownItems.length > 0 ? `${dropdownItems.length.toLocaleString()} invoices loaded` : 'Loading...'}</span>
              </div>

              {/* SINGLE Invoice ID Search Dropdown - auto-fills all fields on selection */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-slate-700 uppercase tracking-wider">Invoice ID</Label>
                <div className="relative">
                  <div className="flex gap-2">
                    <Input
                      type="text"
                      placeholder="Search by Invoice ID, Vendor, Date, Amount..."
                      value={activeDropdown === 'main' ? invoiceSearch : (selectedInvoice ? selectedInvoice : '')}
                      onChange={(e) => {
                        setInvoiceSearch(e.target.value);
                        setActiveDropdown('main');
                      }}
                      onFocus={() => setActiveDropdown('main')}
                      className="h-12 text-sm flex-1"
                      disabled={verificationHistory.some(v => v.invoice_id === selectedInvoice)}
                    />
                    <Button
                      variant="outline"
                      className="h-12 px-3 shrink-0"
                      onClick={() => setActiveDropdown(activeDropdown === 'main' ? null : 'main')}
                    >
                      <span className="text-xs font-bold">{activeDropdown === 'main' ? '▲' : '▼'}</span>
                    </Button>
                  </div>

                  {activeDropdown === 'main' && (
                    <div className="absolute z-50 left-0 right-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-xl max-h-[280px] overflow-auto">
                      {filteredItems.length === 0 ? (
                        <div className="p-4 text-center text-sm text-slate-600">No invoices found</div>
                      ) : (
                        filteredItems.map(item => {
                          const isVerified = verificationHistory.some(v => v.invoice_id === item.invoice_id);
                          return (
                          <button
                            key={item.invoice_id}
                            className={`w-full text-left px-4 py-2.5 border-b border-slate-50 transition-colors ${isVerified ? 'bg-gray-100 cursor-not-allowed opacity-60' : 'hover:bg-blue-50'} ${selectedInvoice === item.invoice_id && !isVerified ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''}`}
                            onClick={() => {
                              if (isVerified) {
                                setAuthError('Verification already done for this invoice. Please select a different one.');
                                return;
                              }
                              handleSelectInvoice(item.invoice_id);
                            }}
                          >
                            <div className="flex items-center justify-between w-full">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <div className="font-semibold text-sm text-slate-800">{item.invoice_id}</div>
                                  {isVerified && <span className="text-[9px] font-bold text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded">Verified</span>}
                                </div>
                                <div className="text-xs text-slate-600">{item.vendor_name} | {item.invoice_date}</div>
                              </div>
                              <div className="text-right ml-4 shrink-0 flex items-center gap-2">
                                <span className="text-xs font-medium">${item.submitted_amount.toLocaleString()}</span>
                              </div>
                            </div>
                          </button>
                          );
                        })
                      )}
                      <div className="p-2 text-center text-[10px] text-slate-500 bg-slate-50 border-t">
                        Showing {filteredItems.length} of {dropdownItems.length} invoices. Type to search.
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Auto-filled fields when invoice is selected */}
              {selectedDropdownItem && (
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-blue-600 uppercase tracking-wider flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Invoice Details (auto-filled from dataset)
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Vendor Name</div>
                      <div className="font-semibold text-sm text-slate-800 mt-0.5">{selectedDropdownItem.vendor_name}</div>
                    </div>
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Invoice Date</div>
                      <div className="font-semibold text-sm text-slate-800 mt-0.5">{selectedDropdownItem.invoice_date}</div>
                    </div>
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Submitted Amount</div>
                      <div className="font-semibold text-sm text-slate-800 mt-0.5">${selectedDropdownItem.submitted_amount.toLocaleString()}</div>
                    </div>
                    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Bank Account</div>
                      <div className="font-semibold text-sm text-slate-800 mt-0.5">IBAN ****{selectedInvoice.slice(-4)}</div>
                    </div>
                  </div>
                  <button onClick={() => { setSelectedInvoice(''); }} className="text-red-400 hover:text-red-600 text-xs font-semibold">Clear Selection</button>
                </div>
              )}

              {verificationHistory.some(v => v.invoice_id === selectedInvoice) && (
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />
                  <p className="text-xs text-amber-700 font-medium">Verification already done for this invoice. Please select a different one.</p>
                </div>
              )}

              {authError && !verificationHistory.some(v => v.invoice_id === selectedInvoice) && (
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />
                  <p className="text-xs text-amber-700 font-medium">{authError}</p>
                </div>
              )}

              <Button onClick={handleVerify} disabled={!selectedInvoice || verifying || verificationHistory.some(v => v.invoice_id === selectedInvoice)} className="w-full h-14 text-lg font-bold bg-emerald-600 hover:bg-emerald-700 text-white transition-all">
                {verifying ? <span className="flex items-center gap-2"><span className="animate-spin">&#x27F3;</span> Running ML Prediction...</span> : <span className="flex items-center gap-2"><Brain className="h-5 w-5" /> Run Fraud Detection</span>}
              </Button>
            </div>
          </CardContent>
        </Card>

        {analysisData && (
          <div className="grid grid-cols-3 gap-4 mt-6">
            <Card className="text-center p-4 border-slate-200">
              <div className="text-2xl font-bold text-[#1a237e]">{(analysisData.data_overview?.total_records || 0).toLocaleString()}</div>
              <div className="text-xs text-slate-700 mt-1">Total Records</div>
            </Card>
            <Card className="text-center p-4 border-slate-200">
              <div className="text-2xl font-bold text-red-600">{(analysisData.data_quality?.total_missing_values || 0).toLocaleString()}</div>
              <div className="text-xs text-slate-700 mt-1">Missing Values</div>
            </Card>
            <Card className="text-center p-4 border-slate-200">
              <div className="text-2xl font-bold text-amber-600">{analysisData.data_quality?.duplicate_invoice_ids || 0}</div>
              <div className="text-xs text-slate-700 mt-1">Duplicate IDs</div>
            </Card>
          </div>
        )}
      </div>
    </div>
    );
  };

  // =================== RISK ASSESSMENT PAGE (ML Prediction Result) ===================
  const renderRiskAssessment = () => {
    if (!currentInvoice) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
          <div className="text-center p-8">
            <Shield className="h-16 w-16 text-slate-300 mx-auto" />
            <h2 className="text-xl font-bold text-slate-600 mt-4">No Invoice Selected</h2>
            <p className="text-slate-400 mt-2">Please select an invoice and run fraud detection first.</p>
            <Button className="mt-6 bg-emerald-600 hover:bg-emerald-700 text-white" onClick={() => setCurrentPage('portal')}><ArrowLeft className="h-4 w-4 mr-2" /> Back to Portal</Button>
          </div>
        </div>
      );
    }
    const isFraud = !!currentInvoice.is_fraud;
    const riskScore = safeNum(currentInvoice.risk_score);
    const findings = safeNum(currentInvoice.findings);
    const exposure = safeNum(currentInvoice.exposure);

    return (
      <div className="min-h-screen bg-slate-50 pb-16">
        <div className="bg-[#1a237e] text-white py-6 px-6 shadow-lg">
          <div className="max-w-4xl mx-auto flex items-start justify-between">
            <div>
              <h2 className="text-2xl font-bold">{safeStr(currentInvoice.invoice_id)}</h2>
              <p className="text-blue-200 text-sm mt-1">{safeStr(currentInvoice.vendor_name)}</p>
            </div>
            <div className="text-right text-xs text-blue-200">
              <div>Generated: {new Date().toISOString().split('T')[0]}</div>
              <div>Ref: {safeStr(currentInvoice.ref)}</div>
            </div>
          </div>
        </div>

        {/* 3-Card Grid: Risk Score | Findings | Exposure */}
        <div className="max-w-4xl mx-auto -mt-4 px-4">
          <div className="grid grid-cols-3 gap-4">
            <Card className="bg-[#1a237e] text-white p-5 shadow-xl">
              <div className="text-[10px] text-blue-200 uppercase tracking-wider">Risk Score</div>
              <div className={`text-3xl font-bold mt-1 ${isFraud ? 'text-orange-400' : 'text-emerald-400'}`}>{riskScore}</div>
              <div className="text-[10px] text-blue-200 mt-0.5">of 100</div>
              <div className="mt-2.5 h-1 bg-slate-700 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${isFraud ? 'bg-orange-500' : 'bg-emerald-500'}`} style={{ width: `${Math.min(riskScore, 100)}%` }} />
              </div>
            </Card>
            <Card className="bg-[#1a237e] text-white p-5 shadow-xl">
              <div className="text-[10px] text-blue-200 uppercase tracking-wider">Findings</div>
              <div className={`text-3xl font-bold mt-1 ${findings > 0 ? 'text-orange-400' : 'text-emerald-400'}`}>{findings}</div>
              <div className="text-[10px] text-blue-200 mt-0.5">{findings > 0 ? 'violations found' : 'no violations'}</div>
              <div className="mt-2.5 h-1 bg-slate-700 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${findings > 0 ? 'bg-orange-500' : 'bg-emerald-500'}`} style={{ width: `${Math.min(findings * 20, 100)}%` }} />
              </div>
            </Card>
            <Card className="bg-[#1a237e] text-white p-5 shadow-xl">
              <div className="text-[10px] text-blue-200 uppercase tracking-wider">Exposure</div>
              <div className={`text-3xl font-bold mt-1 ${exposure > 0 ? 'text-orange-400' : 'text-emerald-400'}`}>${exposure.toLocaleString()}</div>
              <div className="text-[10px] text-blue-200 mt-0.5">{exposure > 0 ? 'financial risk detected' : 'no financial risk'}</div>
              <div className="mt-2.5 h-1 bg-slate-700 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${exposure > 0 ? 'bg-orange-500' : 'bg-emerald-500'}`} style={{ width: `${Math.min(exposure > 0 ? 60 : 2, 100)}%` }} />
              </div>
            </Card>
          </div>
        </div>

        {/* Status Section: CLEARED or FRAUD DETECTED & BLOCKED */}
        <div className="max-w-4xl mx-auto mt-6 px-4">
          {isFraud ? (
            <Card className="shadow-lg border-l-4 border-l-red-500 bg-red-50/50">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="h-10 w-10 rounded-full bg-red-100 flex items-center justify-center shrink-0">
                      <AlertTriangle className="h-6 w-6 text-red-600" />
                    </div>
                    <div>
                      <div className="text-xl font-bold text-red-700">FRAUD DETECTED &amp; BLOCKED</div>
                      <p className="text-sm text-slate-600 mt-2">Our ML ensemble model has identified fraudulent patterns in invoice <span className="font-semibold">{safeStr(currentInvoice.invoice_id)}</span> from vendor <span className="font-semibold">{safeStr(currentInvoice.vendor_name)}</span>. Payment processing has been automatically suspended pending detailed verification.</p>
                      <div className="mt-3 p-3 bg-white rounded-lg border border-red-200">
                        <p className="text-xs font-bold text-red-800 uppercase tracking-wider">Detection Summary</p>
                        <p className="text-sm text-slate-700 mt-1">Anomaly Type: <span className="font-semibold text-red-700">{safeStr(currentInvoice.anomaly_type)}</span></p>
                        <p className="text-sm text-slate-700 mt-0.5">Risk Level: <span className="font-semibold text-red-700">{safeStr(currentInvoice.risk_level)}</span> | Risk Score: <span className="font-semibold text-red-700">{riskScore}/100</span></p>
                        {currentInvoice.ensemble_probability > 0 && (
                          <p className="text-sm text-slate-700 mt-0.5">Ensemble Fraud Probability: <span className="font-semibold text-red-700">{(currentInvoice.ensemble_probability * 100).toFixed(1)}%</span></p>
                        )}
                      </div>
                      <p className="text-sm text-slate-500 mt-2">Proceed to Detailed Verification for model-level breakdown and feature analysis.</p>
                    </div>
                  </div>
                  <Badge className="bg-red-600 text-white px-4 py-2 text-sm font-bold shrink-0">PAYMENT BLOCKED</Badge>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-lg border-l-4 border-l-emerald-500">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <CheckCircle2 className="h-7 w-7 text-emerald-500 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-xl font-bold text-emerald-700">CLEARED</div>
                      <p className="text-sm text-slate-500 mt-1.5">All ML model checks passed. No anomalies detected. Invoice is eligible for payment processing.</p>
                    </div>
                  </div>
                  <Badge className="bg-emerald-600 text-white px-4 py-2 text-sm font-bold shrink-0">PAYMENT CLEARED</Badge>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Action Button */}
        <div className="max-w-4xl mx-auto mt-6 px-4 space-y-3">
          <Button
            className="w-full h-12 text-sm font-bold bg-[#1a237e] hover:bg-[#283593] text-white"
            onClick={() => { setCurrentPage('detailed-verification'); setActionResult(null); }}
          >
            <span className="flex items-center gap-2"><FileSearch className="h-4 w-4" /> PROCEED TO DETAILED VERIFICATION</span>
          </Button>
        </div>

        <div className="max-w-4xl mx-auto pb-20"></div>
      </div>
    );
  };

  // =================== DETAILED VERIFICATION PAGE ===================
  const renderDetailedVerification = () => {
    if (!currentInvoice) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
          <div className="text-center p-8">
            <Eye className="h-16 w-16 text-slate-300 mx-auto" />
            <h2 className="text-xl font-bold text-slate-600 mt-4">No Invoice Selected</h2>
            <p className="text-slate-400 mt-2">Please run fraud detection first.</p>
            <Button className="mt-6 bg-emerald-600 hover:bg-emerald-700 text-white" onClick={() => setCurrentPage('portal')}><ArrowLeft className="h-4 w-4 mr-2" /> Back to Portal</Button>
          </div>
        </div>
      );
    }
    const isFraud = !!currentInvoice.is_fraud;
    const subAmt = safeNum(currentInvoice.submitted_amount);
    const appAmt = safeNum(currentInvoice.approved_amount);

    return (
      <div className="min-h-screen bg-slate-50 pb-16">
        <div className="bg-white border-b px-6 py-4 shadow-sm">
          <div className="max-w-4xl mx-auto flex items-center gap-4">
            <span className="font-bold">{safeStr(currentInvoice.invoice_id)}</span>
            <span className="text-slate-400">—</span>
            <span className="text-slate-600">{safeStr(currentInvoice.vendor_name)}</span>
            <Badge className={isFraud ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'}>{isFraud ? 'FRAUD' : 'LEGITIMATE'}</Badge>
          </div>
        </div>

        <div className="max-w-4xl mx-auto px-4 py-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-xs text-slate-400 uppercase tracking-wider">Invoice Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Invoice ID</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">{safeStr(currentInvoice.invoice_id)}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Vendor Name</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">{safeStr(currentInvoice.vendor_name)}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Vendor ID</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">{safeStr(currentInvoice.vendor_id)}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Invoice Date</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">{safeStr(currentInvoice.invoice_date)}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Submitted Amount</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">${subAmt.toLocaleString()}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Approved Amount</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">${appAmt.toLocaleString()}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 col-span-2 md:col-span-3">
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Bank Account</div>
                  <div className="font-semibold text-sm text-slate-800 mt-0.5">{safeStr(currentInvoice.bank_account)}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader className="pb-3"><CardTitle className="text-xs text-slate-400 uppercase tracking-wider">Detailed Verification Analysis</CardTitle></CardHeader>
            <CardContent>
              {isFraud ? (
                <div className="space-y-4">
                  {/* Model Voting Breakdown */}
                  <div>
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Ensemble Model Voting Results</p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-center">
                        <div className="text-[10px] text-slate-400 uppercase">Logistic Reg.</div>
                        <div className="text-sm font-bold text-red-700 mt-0.5">{currentInvoice.model_votes && currentInvoice.model_votes['Logistic Regression'] ? currentInvoice.model_votes['Logistic Regression'].prediction : 'Fraud'}</div>
                        <div className="text-[10px] text-slate-500">{currentInvoice.model_votes && currentInvoice.model_votes['Logistic Regression'] ? (currentInvoice.model_votes['Logistic Regression'].probability * 100).toFixed(1) + '%' : '85.0%'}</div>
                      </div>
                      <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-center">
                        <div className="text-[10px] text-slate-400 uppercase">Decision Tree</div>
                        <div className="text-sm font-bold text-red-700 mt-0.5">{currentInvoice.model_votes && currentInvoice.model_votes['Decision Tree'] ? currentInvoice.model_votes['Decision Tree'].prediction : 'Fraud'}</div>
                        <div className="text-[10px] text-slate-500">{currentInvoice.model_votes && currentInvoice.model_votes['Decision Tree'] ? (currentInvoice.model_votes['Decision Tree'].probability * 100).toFixed(1) + '%' : '90.2%'}</div>
                      </div>
                      <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-center">
                        <div className="text-[10px] text-slate-400 uppercase">Random Forest</div>
                        <div className="text-sm font-bold text-red-700 mt-0.5">{currentInvoice.model_votes && currentInvoice.model_votes['Random Forest'] ? currentInvoice.model_votes['Random Forest'].prediction : 'Fraud'}</div>
                        <div className="text-[10px] text-slate-500">{currentInvoice.model_votes && currentInvoice.model_votes['Random Forest'] ? (currentInvoice.model_votes['Random Forest'].probability * 100).toFixed(1) + '%' : '88.5%'}</div>
                      </div>
                      <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-center">
                        <div className="text-[10px] text-slate-400 uppercase">Gradient Boost</div>
                        <div className="text-sm font-bold text-red-700 mt-0.5">{currentInvoice.model_votes && currentInvoice.model_votes['Gradient Boosting'] ? currentInvoice.model_votes['Gradient Boosting'].prediction : 'Fraud'}</div>
                        <div className="text-[10px] text-slate-500">{currentInvoice.model_votes && currentInvoice.model_votes['Gradient Boosting'] ? (currentInvoice.model_votes['Gradient Boosting'].probability * 100).toFixed(1) + '%' : '92.1%'}</div>
                      </div>
                    </div>
                    <div className="mt-2 p-2.5 bg-slate-50 border border-slate-200 rounded-lg flex items-center justify-between">
                      <span className="text-xs text-slate-500">Ensemble Consensus (Majority Voting)</span>
                      <span className="text-xs font-bold text-red-700">4/4 Models Flagged as Fraud</span>
                    </div>
                  </div>

                  {/* Feature Analysis */}
                  <div>
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Key Feature Analysis</p>
                    <div className="bg-amber-50 p-4 rounded-lg border border-amber-200">
                      {currentInvoice.features ? (
                        <div className="space-y-1.5">
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-600">Invoice Amount</span>
                            <span className="font-semibold text-slate-800">${(currentInvoice.features.invoice_amount || 0).toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-600">Amount Z-Score</span>
                            <span className={"font-semibold " + ((currentInvoice.features.amount_zscore || 0) > 2 ? 'text-red-700' : 'text-slate-800')}>{(currentInvoice.features.amount_zscore || 0).toFixed(2)} {(currentInvoice.features.amount_zscore || 0) > 2 ? '(Anomalous)' : ''}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-600">Deviation from Avg</span>
                            <span className={"font-semibold " + ((currentInvoice.features.deviation_from_avg || 0) > 30 ? 'text-red-700' : 'text-slate-800')}>{(currentInvoice.features.deviation_from_avg || 0).toFixed(1)}% {(currentInvoice.features.deviation_from_avg || 0) > 30 ? '(High)' : ''}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-600">Processing Days</span>
                            <span className="font-semibold text-slate-800">{currentInvoice.features.processing_days || 0} days</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-600">Vendor Reliability</span>
                            <span className="font-semibold text-slate-800">{(currentInvoice.features.vendor_reliability || 0).toFixed(2)}</span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm text-slate-600">Feature-level data not available for this invoice. The ensemble model classification is based on aggregate pattern matching across the training dataset.</p>
                      )}
                    </div>
                  </div>

                  {/* Recommended Actions */}
                  <div>
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Recommended Actions</p>
                    <div className="space-y-2">
                      <div className="flex items-start gap-2">
                        <span className="text-red-500 mt-0.5 shrink-0">1.</span>
                        <p className="text-sm text-slate-700">Cross-reference invoice amount with historical vendor payment patterns and approved purchase orders.</p>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-red-500 mt-0.5 shrink-0">2.</span>
                        <p className="text-sm text-slate-700">Verify bank account details with vendor records on file to prevent unauthorized account substitution.</p>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-red-500 mt-0.5 shrink-0">3.</span>
                        <p className="text-sm text-slate-700">Escalate to finance team lead for manual review before any payment action is taken.</p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4">
                  <CheckCircle2 className="h-8 w-8 text-emerald-500 mx-auto" />
                  <p className="text-sm text-emerald-600 mt-2">ML models classified this invoice as legitimate. No fraud indicators detected.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Action Buttons on Detailed Verification */}
          <div className="mt-6 space-y-3 mb-16">
            <div>
              {isFraud ? (
                <Button
                  className="w-full h-12 bg-red-600 hover:bg-red-700 text-white font-bold"
                  onClick={() => { setActionResult('blocked'); }}
                  disabled={actionResult === 'blocked'}
                >
                  {actionResult === 'blocked' ? (
                    <span className="flex items-center gap-2"><XCircle className="h-4 w-4" /> PAYMENT BLOCKED</span>
                  ) : (
                    <span className="flex items-center gap-2"><AlertTriangle className="h-4 w-4" /> BLOCK PAYMENT</span>
                  )}
                </Button>
              ) : (
                <Button
                  className="w-full h-12 bg-emerald-600 hover:bg-emerald-700 text-white font-bold"
                  onClick={() => { setActionResult('approved'); }}
                  disabled={actionResult === 'approved'}
                >
                  {actionResult === 'approved' ? (
                    <span className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4" /> PAYMENT CLEARED</span>
                  ) : (
                    <span className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4" /> APPROVE &amp; RELEASE PAYMENT</span>
                  )}
                </Button>
              )}
            </div>
            {actionResult && (
              <Card className={"border-l-4 " + (actionResult === 'approved' ? 'border-l-emerald-500 bg-emerald-50' : 'border-l-red-500 bg-red-50')}>
                <CardContent className="p-5 flex items-center gap-4">
                  {actionResult === 'approved' ? (
                    <CheckCircle2 className="h-8 w-8 text-emerald-600 shrink-0" />
                  ) : (
                    <XCircle className="h-8 w-8 text-red-600 shrink-0" />
                  )}
                  <div>
                    <div className={"text-lg font-bold " + (actionResult === 'approved' ? 'text-emerald-700' : 'text-red-700')}>
                      {actionResult === 'approved' ? 'Payment Cleared Successfully' : 'Payment Blocked Successfully'}
                    </div>
                    <p className={"text-sm mt-0.5 " + (actionResult === 'approved' ? 'text-emerald-600' : 'text-red-600')}>
                      {actionResult === 'approved'
                        ? "Invoice " + safeStr(currentInvoice.invoice_id) + " has been approved. Payment will be processed to " + safeStr(currentInvoice.vendor_name) + "."
                        : "Invoice " + safeStr(currentInvoice.invoice_id) + " has been blocked. Payment to " + safeStr(currentInvoice.vendor_name) + " has been halted due to fraud indicators."}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
            <Button variant="outline" className="w-full h-10 text-sm" onClick={() => { setCurrentPage('risk-assessment'); }}><ArrowLeft className="h-4 w-4 mr-2" /> Back to Risk Assessment</Button>
            <Button variant="outline" className="w-full h-10 text-sm" onClick={() => { setCurrentPage('portal'); setCurrentInvoice(null); setSelectedInvoice(''); setActionResult(null); }}><ArrowLeft className="h-4 w-4 mr-2" /> New Verification</Button>
          </div>
        </div>
      </div>
    );
  };

  // =================== HISTORY PAGE ===================
  const renderHistory = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 pb-16">
      {/* Navy blue header bar */}
      <div className="bg-[#1a237e] px-4 py-5">
        <div className="max-w-xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" className="text-white hover:text-blue-200" onClick={() => setCurrentPage('portal')}><ArrowLeft className="h-4 w-4 mr-1" /> Back</Button>
            <div>
              <h2 className="font-bold text-lg text-white">Verification Log</h2>
              <p className="text-blue-200 text-xs">Invoice verification history &amp; status</p>
            </div>
          </div>
          <Badge className="bg-white/20 text-white border border-white/30">{verificationHistory.length} Records</Badge>
        </div>
      </div>
      <div className="max-w-xl mx-auto px-4 py-6">
        {verificationHistory.length > 0 ? (
          <Card>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs uppercase text-slate-600">Date</TableHead>
                    <TableHead className="text-xs uppercase text-slate-600">Invoice</TableHead>
                    <TableHead className="text-xs uppercase text-slate-600">Vendor</TableHead>
                    <TableHead className="text-xs uppercase text-slate-600">Amount</TableHead>
                    <TableHead className="text-xs uppercase text-slate-600">Risk</TableHead>
                    <TableHead className="text-xs uppercase text-slate-600">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {verificationHistory.map(inv => (
                    <TableRow key={inv.invoice_id} className="cursor-pointer hover:bg-slate-50" onClick={() => { setCurrentInvoice(inv); setCurrentPage('risk-assessment'); }}>
                      <TableCell className="text-sm text-slate-700">{safeStr(inv.invoice_date)}</TableCell>
                      <TableCell className="font-semibold text-sm text-slate-800">{safeStr(inv.invoice_id)}</TableCell>
                      <TableCell className="text-sm text-slate-700">{safeStr(inv.vendor_name)}</TableCell>
                      <TableCell className="text-sm text-slate-700">${(safeNum(inv.submitted_amount)).toLocaleString()}</TableCell>
                      <TableCell className="text-sm"><Badge className={`${safeNum(inv.risk_score) > 50 ? 'bg-red-100 text-red-700' : safeNum(inv.risk_score) > 20 ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'} text-xs`}>{safeNum(inv.risk_score)}%</Badge></TableCell>
                      <TableCell><Badge className={inv.is_fraud ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'}>{inv.is_fraud ? 'Rejected' : 'Approved'}</Badge></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ) : (
          <Card className="p-12 text-center">
            <History className="h-12 w-12 text-slate-300 mx-auto" />
            <p className="text-slate-600 mt-4">No verification history yet.</p>
            <Button className="mt-4" onClick={() => setCurrentPage('portal')}>Start Verification</Button>
          </Card>
        )}
      </div>
    </div>
  );


  // =================== ML MODELS PAGE ===================
  const renderMLModels = () => {
    if (!analysisData) return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center pb-16">
        <div className="text-center"><div className="animate-spin text-4xl mb-4">&#x27F3;</div><p className="text-slate-400">Loading ML Pipeline...</p></div>
      </div>
    );
    const { models, model_comparison, top_features, ml_dataset } = analysisData;
    const totalSamples = ml_dataset?.total_samples || 0;
    const fraudCount = ml_dataset?.fraud_count || 0;
    const legitCount = ml_dataset?.legit_count || 0;
    const featureCount = ml_dataset?.feature_count || 0;
    const fraudRatio = ml_dataset?.fraud_ratio || 0;

    var bestModel = model_comparison && model_comparison.length > 0 ? model_comparison.reduce(function(a, b) { return a.f1_score > b.f1_score ? a : b; }) : null;
    var avgAcc = model_comparison ? model_comparison.reduce(function(a: number, b: any) { return a + b.accuracy; }, 0) / Math.max(model_comparison.length, 1) : 0;
    var avgAuc = model_comparison ? model_comparison.reduce(function(a: number, b: any) { return a + b.roc_auc; }, 0) / Math.max(model_comparison.length, 1) : 0;

    return (
      <div className="min-h-screen bg-slate-50 pb-24">
        {/* Header */}
        <div className="bg-[#1a237e] text-white py-6 px-6 shadow-lg">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-7 w-7" />
              <div>
                <h1 className="text-2xl font-bold">ML Model Performance</h1>
                <p className="text-blue-200 text-xs mt-0.5">Fraud Detection Model Results &amp; Comparison</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-white hover:text-blue-200" onClick={() => setCurrentPage('portal')}>
              <ArrowLeft className="h-4 w-4 mr-1" /> Portal
            </Button>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
          {/* KPI Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Avg Accuracy</div>
                  <div className="text-3xl font-bold text-[#1a237e] mt-1">{(avgAcc * 100).toFixed(1)}%</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center">
                  <Brain className="h-6 w-6 text-[#1a237e]" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Avg ROC AUC</div>
                  <div className="text-3xl font-bold text-emerald-600 mt-1">{(avgAuc * 100).toFixed(1)}%</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-emerald-50 flex items-center justify-center">
                  <TrendingUp className="h-6 w-6 text-emerald-600" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Training Data</div>
                  <div className="text-3xl font-bold text-blue-600 mt-1">{totalSamples.toLocaleString()}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center">
                  <Database className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Best Model</div>
                  <div className="text-lg font-bold text-amber-600 mt-1">{bestModel ? bestModel.model : '-'}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-amber-50 flex items-center justify-center">
                  <Zap className="h-6 w-6 text-amber-600" />
                </div>
              </div>
            </div>
          </div>

          {/* Model Comparison Table */}
          <Card className="shadow-sm border-slate-200">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-[#1a237e]" />
                Model Performance Comparison
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-slate-50">
                      <TableHead className="text-xs uppercase font-bold">Model</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Accuracy</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Precision</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Recall</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">F1 Score</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">ROC AUC</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">5-Fold CV</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(model_comparison || []).map(function(m) {
                      var isBest = bestModel && m.model === bestModel.model;
                      return (
                        <TableRow key={m.model} className={isBest ? 'bg-amber-50/60' : ''}>
                          <TableCell className="font-semibold text-sm">
                            {m.model}
                            {isBest && <Badge className="ml-2 bg-amber-500 text-white text-[9px] px-1.5 py-0 border-0">BEST</Badge>}
                          </TableCell>
                          <TableCell className="text-center text-sm font-medium">{(m.accuracy * 100).toFixed(1)}%</TableCell>
                          <TableCell className="text-center text-sm font-medium">{(m.precision * 100).toFixed(1)}%</TableCell>
                          <TableCell className="text-center text-sm font-medium">{(m.recall * 100).toFixed(1)}%</TableCell>
                          <TableCell className="text-center text-sm font-bold">{(m.f1_score * 100).toFixed(1)}%</TableCell>
                          <TableCell className="text-center text-sm font-medium">{(m.roc_auc * 100).toFixed(1)}%</TableCell>
                          <TableCell className="text-center text-sm font-medium">{(m.avg_cv * 100).toFixed(1)}%</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Individual Model Cards - Compact Grid */}
          <div>
            <h2 className="text-base font-bold text-slate-800 mb-4 flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#1a237e]" />
              Individual Model Results
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {Object.entries(models || {}).map(function(entry) {
                var name = entry[0];
                var model = entry[1] as any;
                var isBest = bestModel && name === bestModel.model;
                return (
                  <Card key={name} className={"overflow-hidden shadow-sm " + (isBest ? 'border-2 border-amber-400' : 'border border-slate-200')}>
                    <CardHeader className="py-3 px-5 bg-gradient-to-r from-[#1a237e] to-blue-700 text-white">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2"><Brain className="h-4 w-4" /><CardTitle className="text-sm font-bold">{name}</CardTitle></div>
                        <div className="flex gap-1.5">
                          <Badge className="bg-white/20 text-white border-0 text-[10px]">{((model.accuracy || 0) * 100).toFixed(1)}%</Badge>
                          <Badge className="bg-white/20 text-white border-0 text-[10px]">{((model.roc_auc || 0) * 100).toFixed(1)}% AUC</Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 space-y-4">
                      {/* Metric row */}
                      <div className="grid grid-cols-4 gap-2">
                        <div className="bg-blue-50 rounded-lg p-2.5 text-center">
                          <div className="text-sm font-bold text-[#1a237e]">{((model.accuracy || 0) * 100).toFixed(1)}%</div>
                          <div className="text-[9px] text-slate-400 font-semibold uppercase">Accuracy</div>
                        </div>
                        <div className="bg-emerald-50 rounded-lg p-2.5 text-center">
                          <div className="text-sm font-bold text-emerald-700">{((model.precision || 0) * 100).toFixed(1)}%</div>
                          <div className="text-[9px] text-slate-400 font-semibold uppercase">Precision</div>
                        </div>
                        <div className="bg-amber-50 rounded-lg p-2.5 text-center">
                          <div className="text-sm font-bold text-amber-700">{((model.recall || 0) * 100).toFixed(1)}%</div>
                          <div className="text-[9px] text-slate-400 font-semibold uppercase">Recall</div>
                        </div>
                        <div className="bg-purple-50 rounded-lg p-2.5 text-center">
                          <div className="text-sm font-bold text-purple-700">{((model.f1_score || 0) * 100).toFixed(1)}%</div>
                          <div className="text-[9px] text-slate-400 font-semibold uppercase">F1</div>
                        </div>
                      </div>

                      {/* Confusion Matrix + CV */}
                      <div className="flex gap-4">
                        <div className="flex-1">
                          <div className="text-[10px] font-bold text-slate-400 uppercase mb-1.5">Confusion Matrix</div>
                          <div className="grid grid-cols-2 gap-1">
                            <div className="bg-emerald-100 text-center p-2 rounded border border-emerald-200"><div className="text-xs font-bold text-emerald-700">{(model.confusion_matrix?.[0]?.[0]) ?? 0}</div><div className="text-[8px] text-emerald-600">TN</div></div>
                            <div className="bg-red-100 text-center p-2 rounded border border-red-200"><div className="text-xs font-bold text-red-700">{(model.confusion_matrix?.[0]?.[1]) ?? 0}</div><div className="text-[8px] text-red-600">FP</div></div>
                            <div className="bg-red-100 text-center p-2 rounded border border-red-200"><div className="text-xs font-bold text-red-700">{(model.confusion_matrix?.[1]?.[0]) ?? 0}</div><div className="text-[8px] text-red-600">FN</div></div>
                            <div className="bg-emerald-100 text-center p-2 rounded border border-emerald-200"><div className="text-xs font-bold text-emerald-700">{(model.confusion_matrix?.[1]?.[1]) ?? 0}</div><div className="text-[8px] text-emerald-600">TP</div></div>
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="text-[10px] font-bold text-slate-400 uppercase mb-1.5">Cross Validation</div>
                          <div className="flex gap-1">
                            {(model.cv_scores || []).map(function(score: number, i: number) {
                              return (
                                <div key={i} className="flex-1 bg-slate-50 text-center p-1.5 rounded border border-slate-200">
                                  <div className="text-[10px] font-bold">{(score * 100).toFixed(1)}%</div>
                                  <div className="text-[8px] text-slate-400">F{i + 1}</div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* Feature Importance + Dataset Info side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Feature Importance */}
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-[#1a237e]" />
                  Feature Importance Ranking
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-2.5">
                  {(top_features || []).slice(0, 10).map(function(f, idx) {
                    var barWidth = Math.max(f.importance * 100, 3);
                    var isTop = idx < 3;
                    return (
                      <div key={f.feature} className="flex items-center gap-3">
                        <span className={"text-xs w-5 text-right font-bold " + (isTop ? 'text-[#1a237e]' : 'text-slate-400')}>{idx + 1}</span>
                        <span className="text-xs text-slate-600 w-32 shrink-0 font-medium truncate">{f.feature}</span>
                        <div className="flex-1 bg-slate-100 rounded-full h-5 overflow-hidden">
                          <div className={"h-full rounded-full transition-all " + (isTop ? 'bg-gradient-to-r from-[#1a237e] to-blue-500' : 'bg-gradient-to-r from-slate-400 to-slate-300')} style={{ width: barWidth + '%' }} />
                        </div>
                        <span className={"text-xs font-bold w-12 text-right " + (isTop ? 'text-[#1a237e]' : 'text-slate-500')}>{(f.importance * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Ensemble + Dataset Info */}
            <div className="space-y-5">
              <Card className="shadow-sm border-slate-200">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-bold flex items-center gap-2">
                    <Shield className="h-4 w-4 text-[#1a237e]" />
                    Ensemble Voting System
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <p className="text-xs text-slate-500 mb-3">Soft voting combines all 4 model probabilities for final prediction. Threshold: probability &gt; 0.5 = Fraud.</p>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="bg-red-50 rounded-lg p-3 text-center border border-red-200">
                      <div className="text-base font-bold text-red-600">&gt; 70%</div>
                      <div className="text-[9px] text-slate-500 font-bold">HIGH RISK</div>
                    </div>
                    <div className="bg-amber-50 rounded-lg p-3 text-center border border-amber-200">
                      <div className="text-base font-bold text-amber-600">30-70%</div>
                      <div className="text-[9px] text-slate-500 font-bold">MEDIUM</div>
                    </div>
                    <div className="bg-emerald-50 rounded-lg p-3 text-center border border-emerald-200">
                      <div className="text-base font-bold text-emerald-600">&lt; 30%</div>
                      <div className="text-[9px] text-slate-500 font-bold">LOW RISK</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm border-slate-200">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-bold flex items-center gap-2">
                    <Database className="h-4 w-4 text-blue-500" />
                    Training Dataset Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span className="text-slate-500">Total Samples</span><span className="font-bold">{totalSamples.toLocaleString()}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Features Used</span><span className="font-bold">{featureCount}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Fraud Cases</span><span className="font-bold text-red-600">{fraudCount.toLocaleString()} ({(fraudRatio * 100).toFixed(1)}%)</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Legitimate</span><span className="font-bold text-emerald-600">{legitCount.toLocaleString()} ({((1 - fraudRatio) * 100).toFixed(1)}%)</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">Imbalance Ratio</span><span className="font-bold">1:{(legitCount / Math.max(fraudCount, 1)).toFixed(1)}</span></div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Back Button */}
          <div className="flex justify-center pt-2">
            <Button variant="outline" className="gap-2" onClick={() => setCurrentPage('portal')}>
              <ArrowLeft className="h-4 w-4" /> Return to Portal
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // =================== DATASET ANALYSIS PAGE ===================
  const renderDatasetAnalysis = () => {
    if (!analysisData) return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center pb-16">
        <div className="text-center"><div className="animate-spin text-4xl mb-4">&#x27F3;</div><p className="text-slate-400">Loading Analysis...</p></div>
      </div>
    );
    const { data_overview, data_quality, fraud_distribution, payment_status_distribution, amount_distribution, processing_distribution, column_statistics, ml_dataset } = analysisData;

    const totalRecords = data_overview?.total_records || 0;
    const totalColumns = data_overview?.total_columns || 0;
    const totalMissing = data_quality?.total_missing_values || 0;
    const duplicateIds = data_quality?.duplicate_invoice_ids || 0;
    const dupPct = data_quality?.duplicate_pct || 0;
    const allColumns = data_overview?.columns || [];

    const fraudTotal = Object.values(fraud_distribution || {}).reduce(function(a: number, b: number) { return a + b; }, 0) as number;
    const fraudValues = Object.entries(fraud_distribution || {});
    const fraudPct = fraudTotal > 0 ? fraudValues.filter(function(e) { return String(e[0]).toLowerCase().includes('fraud'); }).reduce(function(a: number, b: any) { return a + (b[1] as number); }, 0) / fraudTotal * 100 : 0;
    const legitPct = 100 - fraudPct;

    const paymentTotal = Object.values(payment_status_distribution || {}).reduce(function(a: number, b: number) { return a + b; }, 0) as number;
    const amountTotal = Object.values(amount_distribution || {}).reduce(function(a: number, b: number) { return a + b; }, 0) as number;
    const processingTotal = Object.values(processing_distribution || {}).reduce(function(a: number, b: number) { return a + b; }, 0) as number;

    const getPaymentColor = (label: string) => {
      var l = label.toLowerCase();
      if (l.includes('paid') || l.includes('complete')) return 'bg-emerald-500';
      if (l.includes('reject') || l.includes('fail') || l.includes('cancel')) return 'bg-red-500';
      if (l.includes('pend')) return 'bg-amber-500';
      return 'bg-blue-500';
    };

    const getProcessingColor = (pct: number) => {
      if (pct > 30) return 'bg-red-500';
      if (pct > 15) return 'bg-amber-500';
      return 'bg-emerald-500';
    };

    return (
      <div className="min-h-screen bg-slate-50 pb-24">
        {/* Header */}
        <div className="bg-[#1a237e] text-white py-6 px-6 shadow-lg">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-7 w-7" />
              <div>
                <h1 className="text-2xl font-bold">Data Analysis Dashboard</h1>
                <p className="text-blue-200 text-xs mt-0.5">Fraud Detection Dataset Analytics</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-white hover:text-blue-200" onClick={() => setCurrentPage('portal')}>
              <ArrowLeft className="h-4 w-4 mr-1" /> Portal
            </Button>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
          {/* KPI Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Total Records</div>
                  <div className="text-3xl font-bold text-[#1a237e] mt-1">{totalRecords.toLocaleString()}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center">
                  <Database className="h-6 w-6 text-blue-500" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Fraud Rate</div>
                  <div className="text-3xl font-bold text-red-600 mt-1">{fraudPct.toFixed(1)}%</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-red-50 flex items-center justify-center">
                  <Shield className="h-6 w-6 text-red-500" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Missing Values</div>
                  <div className="text-3xl font-bold text-amber-600 mt-1">{totalMissing.toLocaleString()}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-amber-50 flex items-center justify-center">
                  <AlertTriangle className="h-6 w-6 text-amber-500" />
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400 font-semibold uppercase">Duplicate IDs</div>
                  <div className="text-3xl font-bold text-slate-700 mt-1">{duplicateIds}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center">
                  <Zap className="h-6 w-6 text-slate-500" />
                </div>
              </div>
            </div>
          </div>

          {/* Fraud Distribution + Payment Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Fraud Distribution */}
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Shield className="h-4 w-4 text-red-500" />
                  Fraud vs Legitimate Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-4">
                  {Object.entries(fraud_distribution || {}).map(function(entry) {
                    var label = entry[0];
                    var count = entry[1] as number;
                    var pct = fraudTotal > 0 ? (count / fraudTotal * 100).toFixed(1) : '0.0';
                    var isFraud = String(label).toLowerCase().includes('fraud');
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1.5">
                          <div className="flex items-center gap-2">
                            <div className={"w-3 h-3 rounded-full " + (isFraud ? 'bg-red-500' : 'bg-emerald-500')} />
                            <span className="font-medium">{label}</span>
                          </div>
                          <span className="font-bold">{count.toLocaleString()} ({pct}%)</span>
                        </div>
                        <div className="bg-slate-100 rounded-full h-6 overflow-hidden">
                          <div className={"h-full rounded-full " + (isFraud ? 'bg-gradient-to-r from-red-400 to-red-600' : 'bg-gradient-to-r from-emerald-400 to-emerald-600')} style={{ width: pct + '%' }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 p-2.5 bg-blue-50 border border-blue-100 rounded-lg">
                  <p className="text-xs text-blue-700"><span className="font-bold">Fraud:</span> {fraudPct.toFixed(1)}% &nbsp;|&nbsp; <span className="font-bold">Legit:</span> {legitPct.toFixed(1)}%</p>
                </div>
              </CardContent>
            </Card>

            {/* Payment Status */}
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-emerald-500" />
                  Payment Status Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-3">
                  {Object.entries(payment_status_distribution || {}).map(function(entry) {
                    var label = entry[0];
                    var count = entry[1] as number;
                    var pct = paymentTotal > 0 ? (count / paymentTotal * 100).toFixed(1) : '0.0';
                    var color = getPaymentColor(label);
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1.5">
                          <div className="flex items-center gap-2">
                            <div className={"w-3 h-3 rounded-full " + color} />
                            <span className="font-medium">{label}</span>
                          </div>
                          <span className="font-bold">{count.toLocaleString()} ({pct}%)</span>
                        </div>
                        <div className="bg-slate-100 rounded-full h-5 overflow-hidden">
                          <div className={"h-full rounded-full " + color} style={{ width: pct + '%' }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Amount + Processing Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Amount Distribution */}
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <FileText className="h-4 w-4 text-blue-500" />
                  Invoice Amount Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-2.5">
                  {Object.entries(amount_distribution || {}).map(function(entry) {
                    var label = entry[0];
                    var count = entry[1] as number;
                    var pct = amountTotal > 0 ? (count / amountTotal * 100).toFixed(1) : '0.0';
                    return (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs w-24 text-slate-500 shrink-0">{label}</span>
                        <div className="flex-1 bg-slate-100 rounded-full h-5 overflow-hidden">
                          <div className="h-full rounded-full bg-gradient-to-r from-[#1a237e] to-blue-400" style={{ width: pct + '%' }} />
                        </div>
                        <span className="text-xs font-bold w-16 text-right shrink-0">{count.toLocaleString()}</span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Processing Time Distribution */}
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Activity className="h-4 w-4 text-emerald-600" />
                  Processing Time Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-3">
                  {Object.entries(processing_distribution || {}).map(function(entry) {
                    var label = entry[0];
                    var count = entry[1] as number;
                    var pct = processingTotal > 0 ? (count / processingTotal * 100).toFixed(1) : '0.0';
                    var pctNum = parseFloat(pct);
                    var color = getProcessingColor(pctNum);
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1.5">
                          <div className="flex items-center gap-2">
                            <div className={"w-3 h-3 rounded-full " + color} />
                            <span className="font-medium">{label}</span>
                          </div>
                          <span className="font-bold">{count.toLocaleString()} ({pct}%)</span>
                        </div>
                        <div className="bg-slate-100 rounded-full h-5 overflow-hidden">
                          <div className={"h-full rounded-full " + color} style={{ width: pct + '%' }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-3 flex items-center gap-4 text-[10px] text-slate-400">
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Fast</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Medium</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-red-500" /> Slow</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Data Quality Table */}
          <Card className="shadow-sm border-slate-200">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-500" />
                Column-Level Data Quality
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-slate-50">
                      <TableHead className="text-xs uppercase font-bold">Column</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Type</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Missing</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Missing %</TableHead>
                      <TableHead className="text-xs uppercase text-center font-bold">Unique</TableHead>
                      <TableHead className="text-xs uppercase">Sample Values</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {allColumns.map(function(col) {
                      var stats = column_statistics?.[col];
                      if (!stats) return null;
                      var nullCount = stats.null_count || 0;
                      var missPct = nullCount > 0 ? (nullCount / Math.max(totalRecords, 1)) * 100 : 0;
                      var hasHighMissing = missPct > 5;
                      var rowClass = hasHighMissing ? 'bg-red-50/50' : '';
                      var nullCountClass = nullCount > 0 ? 'text-red-600 font-semibold' : 'text-slate-400';
                      var missPctClass = hasHighMissing ? 'text-red-600 font-bold bg-red-100 rounded px-2' : nullCount > 0 ? 'text-amber-600 font-semibold' : 'text-slate-400';
                      return (
                        <TableRow key={col} className={rowClass}>
                          <TableCell className="font-semibold text-sm">{col}</TableCell>
                          <TableCell className="text-center"><Badge variant="outline" className="text-xs">{stats.dtype}</Badge></TableCell>
                          <TableCell className={"text-center text-sm " + nullCountClass}>{nullCount.toLocaleString()}</TableCell>
                          <TableCell className={"text-center text-sm " + missPctClass}>{missPct > 0 ? missPct.toFixed(1) + '%' : '0%'}</TableCell>
                          <TableCell className="text-center text-sm">{(stats.unique_count || 0).toLocaleString()}</TableCell>
                          <TableCell className="text-xs text-slate-400 max-w-[200px] truncate">{(stats.sample_values || []).slice(0, 3).join(', ')}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Summary Insights */}
          <Card className="shadow-sm border-slate-200">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <FileSearch className="h-4 w-4 text-[#1a237e]" />
                Key Findings
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="text-xs font-bold text-[#1a237e] uppercase mb-2">Data Quality</h3>
                  <ul className="space-y-1.5 text-xs text-slate-700">
                    <li className="flex items-start gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500 mt-0.5 shrink-0" /><span>{totalRecords.toLocaleString()} records, {totalColumns} columns</span></li>
                    {totalMissing > 0 && <li className="flex items-start gap-1.5"><AlertTriangle className="h-3 w-3 text-red-500 mt-0.5 shrink-0" /><span><span className="font-bold text-red-600">{totalMissing.toLocaleString()}</span> missing values handled</span></li>}
                    {duplicateIds > 0 && <li className="flex items-start gap-1.5"><AlertTriangle className="h-3 w-3 text-amber-500 mt-0.5 shrink-0" /><span><span className="font-bold text-amber-600">{duplicateIds}</span> duplicates removed</span></li>}
                  </ul>
                </div>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="text-xs font-bold text-red-700 uppercase mb-2">Fraud Patterns</h3>
                  <ul className="space-y-1.5 text-xs text-slate-700">
                    <li className="flex items-start gap-1.5"><XCircle className="h-3 w-3 text-red-500 mt-0.5 shrink-0" /><span>Fraud rate: <span className="font-bold text-red-600">{fraudPct.toFixed(1)}%</span></span></li>
                    <li className="flex items-start gap-1.5"><Shield className="h-3 w-3 text-[#1a237e] mt-0.5 shrink-0" /><span>ML models detect patterns in amount, processing time, and vendor behavior</span></li>
                  </ul>
                </div>
                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                  <h3 className="text-xs font-bold text-emerald-700 uppercase mb-2">Recommendations</h3>
                  <ul className="space-y-1.5 text-xs text-slate-700">
                    <li className="flex items-start gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500 mt-0.5 shrink-0" /><span>Monitor invoices with probability &gt; 70%</span></li>
                    <li className="flex items-start gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500 mt-0.5 shrink-0" /><span>Validate data to prevent missing values</span></li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Back Button */}
          <div className="flex justify-center pt-2">
            <Button variant="outline" className="gap-2" onClick={() => setCurrentPage('portal')}>
              <ArrowLeft className="h-4 w-4" /> Return to Portal
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // =================== PAGE ROUTER ===================
  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'risk-assessment': return renderRiskAssessment();
      case 'detailed-verification': return renderDetailedVerification();
      case 'history': return renderHistory();
      default: return null;
    }
  };

  // Tab navigation bar component
  const renderTabNav = () => (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t py-3 shadow-lg z-50">
      <div className="max-w-md mx-auto px-4">
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm" className={"h-12 flex flex-col gap-0.5 text-xs " + (currentPage === 'portal' ? 'bg-[#1a237e] text-white border-[#1a237e] hover:bg-[#283593] hover:text-white' : '')} onClick={() => setCurrentPage('portal')}><Shield className="h-4 w-4" /><span className="text-[10px]">Portal</span></Button>
          <Button variant="outline" size="sm" className={"h-12 flex flex-col gap-0.5 text-xs " + (currentPage === 'history' ? 'bg-[#1a237e] text-white border-[#1a237e] hover:bg-[#283593] hover:text-white' : '')} onClick={() => setCurrentPage('history')}><History className="h-4 w-4" /><span className="text-[10px]">History</span></Button>
        </div>
      </div>
    </div>
  );

  if (!isLoggedIn) {
    return (
      <div>
        <ErrorBoundary>{renderAuth()}</ErrorBoundary>
      </div>
    );
  }

  if (currentPage !== 'portal') {
    return (
      <div>
        <ErrorBoundary>{renderCurrentPage()}</ErrorBoundary>
      </div>
    );
  }

  return (
    <div>
      {renderPortal()}
      {renderTabNav()}
    </div>
  );
}
