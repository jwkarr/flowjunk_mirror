# Code Samples

> Sanitized code examples demonstrating patterns and architecture used in FlowJunk.

---

## Table of Contents

1. [TypeScript Types & Interfaces](#typescript-types--interfaces)
2. [React Hooks](#react-hooks)
3. [React Components](#react-components)
4. [Supabase Edge Functions](#supabase-edge-functions)
5. [Database Patterns (RLS)](#database-patterns-rls)
6. [AI Integration](#ai-integration)

---

## TypeScript Types & Interfaces

### Quote Item Type Definition

```typescript
// types/quote.ts
export interface QuoteItem {
  id: string;
  name: string;
  category: 'furniture' | 'appliance' | 'electronics' | 'debris' | 'other';
  quantity: number;
  basePrice: number;
  volumeCubicYards: number;
  confidence?: number; // AI detection confidence (0-1)
  source: 'manual' | 'ai-detected' | 'imported';
}

export interface Quote {
  id: string;
  customerId: string;
  items: QuoteItem[];
  subtotal: number;
  accessFactors: AccessFactor[];
  discounts: Discount[];
  truckLoadTier: TruckLoadTier;
  finalTotal: number;
  status: 'draft' | 'sent' | 'accepted' | 'rejected' | 'expired';
  createdAt: Date;
  expiresAt: Date;
}

export interface AccessFactor {
  id: string;
  name: string;
  type: 'stairs' | 'elevator' | 'distance' | 'heavy_item' | 'disassembly';
  rate: number;
  unitType: 'flat' | 'per_flight' | 'per_item' | 'percentage';
}

export type TruckLoadTier = 
  | '1/8' | '1/4' | '3/8' | '1/2' 
  | '5/8' | '3/4' | '7/8' | 'full';
```

### Pricing Configuration Schema (Zod)

```typescript
// schemas/pricing.ts
import { z } from 'zod';

export const PricingItemSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(100),
  category: z.enum(['furniture', 'appliance', 'electronics', 'debris', 'other']),
  basePrice: z.number().min(0),
  volumeCubicYards: z.number().min(0),
  isActive: z.boolean().default(true),
  keywords: z.array(z.string()).optional(),
});

export const TruckLoadSchema = z.object({
  tier: z.enum(['1/8', '1/4', '3/8', '1/2', '5/8', '3/4', '7/8', 'full']),
  minimumPrice: z.number().min(0),
  volumeRange: z.object({
    min: z.number(),
    max: z.number(),
  }),
});

export const PricingConfigSchema = z.object({
  items: z.array(PricingItemSchema),
  truckLoads: z.array(TruckLoadSchema),
  regionalMultiplier: z.number().min(0.5).max(2.0).default(1.0),
  updatedAt: z.date(),
});

export type PricingItem = z.infer<typeof PricingItemSchema>;
export type PricingConfig = z.infer<typeof PricingConfigSchema>;
```

---

## React Hooks

### useQuoteCalculation Hook

```typescript
// hooks/useQuoteCalculation.ts
import { useMemo } from 'react';
import type { QuoteItem, AccessFactor, Discount, TruckLoadTier } from '@/types/quote';

interface CalculationResult {
  itemsTotal: number;
  accessFactorsTotal: number;
  discountsTotal: number;
  truckMinimum: number;
  subtotal: number;
  finalTotal: number;
  appliedTier: TruckLoadTier;
}

export function useQuoteCalculation(
  items: QuoteItem[],
  accessFactors: AccessFactor[],
  discounts: Discount[],
  truckLoadPricing: Record<TruckLoadTier, number>
): CalculationResult {
  return useMemo(() => {
    // Calculate items total
    const itemsTotal = items.reduce(
      (sum, item) => sum + item.basePrice * item.quantity,
      0
    );

    // Calculate total volume to determine truck tier
    const totalVolume = items.reduce(
      (sum, item) => sum + item.volumeCubicYards * item.quantity,
      0
    );

    // Determine applicable truck load tier
    const appliedTier = determineTruckTier(totalVolume);
    const truckMinimum = truckLoadPricing[appliedTier] || 0;

    // Calculate access factors
    const accessFactorsTotal = accessFactors.reduce((sum, factor) => {
      if (factor.unitType === 'percentage') {
        return sum + (itemsTotal * factor.rate) / 100;
      }
      return sum + factor.rate;
    }, 0);

    // Subtotal before discounts (max of items or truck minimum)
    const subtotal = Math.max(itemsTotal, truckMinimum) + accessFactorsTotal;

    // Calculate discounts
    const discountsTotal = discounts.reduce((sum, discount) => {
      if (discount.type === 'percentage') {
        return sum + (subtotal * discount.value) / 100;
      }
      return sum + discount.value;
    }, 0);

    const finalTotal = Math.max(0, subtotal - discountsTotal);

    return {
      itemsTotal,
      accessFactorsTotal,
      discountsTotal,
      truckMinimum,
      subtotal,
      finalTotal,
      appliedTier,
    };
  }, [items, accessFactors, discounts, truckLoadPricing]);
}

function determineTruckTier(volumeCubicYards: number): TruckLoadTier {
  if (volumeCubicYards <= 2) return '1/8';
  if (volumeCubicYards <= 4) return '1/4';
  if (volumeCubicYards <= 6) return '3/8';
  if (volumeCubicYards <= 8) return '1/2';
  if (volumeCubicYards <= 10) return '5/8';
  if (volumeCubicYards <= 12) return '3/4';
  if (volumeCubicYards <= 14) return '7/8';
  return 'full';
}
```

### useCredits Hook (Usage Tracking)

```typescript
// hooks/useCredits.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useCompany } from '@/hooks/useCompany';
import { toast } from 'sonner';

interface CreditBalance {
  visionCredits: { used: number; limit: number };
  voiceMinutes: { used: number; limit: number };
  automationCredits: { used: number; limit: number };
  resetAt: Date;
}

export function useCredits() {
  const { companyId } = useCompany();
  const queryClient = useQueryClient();

  const { data: balance, isLoading } = useQuery({
    queryKey: ['credits', companyId],
    queryFn: async (): Promise<CreditBalance> => {
      const { data, error } = await supabase
        .from('company_subscriptions')
        .select(`
          vision_credits_used,
          vision_credits_limit,
          voice_minutes_used,
          voice_minutes_limit,
          automation_credits_used,
          automation_credits_limit,
          credits_reset_at
        `)
        .eq('company_id', companyId)
        .single();

      if (error) throw error;

      return {
        visionCredits: {
          used: data.vision_credits_used ?? 0,
          limit: data.vision_credits_limit ?? 0,
        },
        voiceMinutes: {
          used: data.voice_minutes_used ?? 0,
          limit: data.voice_minutes_limit ?? 0,
        },
        automationCredits: {
          used: data.automation_credits_used ?? 0,
          limit: data.automation_credits_limit ?? 0,
        },
        resetAt: new Date(data.credits_reset_at),
      };
    },
    enabled: !!companyId,
  });

  const consumeCredit = useMutation({
    mutationFn: async ({ 
      type, 
      amount 
    }: { 
      type: 'vision' | 'voice' | 'automation'; 
      amount: number 
    }) => {
      const { error } = await supabase.rpc('consume_credits', {
        p_company_id: companyId,
        p_credit_type: type,
        p_amount: amount,
      });

      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['credits', companyId] });
    },
    onError: (error) => {
      toast.error('Failed to consume credits', {
        description: error.message,
      });
    },
  });

  const hasCredits = (type: 'vision' | 'voice' | 'automation'): boolean => {
    if (!balance) return false;
    
    switch (type) {
      case 'vision':
        return balance.visionCredits.used < balance.visionCredits.limit;
      case 'voice':
        return balance.voiceMinutes.used < balance.voiceMinutes.limit;
      case 'automation':
        return balance.automationCredits.used < balance.automationCredits.limit;
    }
  };

  return {
    balance,
    isLoading,
    consumeCredit,
    hasCredits,
  };
}
```

---

## React Components

### QuoteSummaryCard Component

```tsx
// components/quotes/QuoteSummaryCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { formatCurrency } from '@/lib/utils';
import type { CalculationResult } from '@/hooks/useQuoteCalculation';

interface QuoteSummaryCardProps {
  calculation: CalculationResult;
  itemCount: number;
  className?: string;
}

export function QuoteSummaryCard({ 
  calculation, 
  itemCount,
  className 
}: QuoteSummaryCardProps) {
  const {
    itemsTotal,
    accessFactorsTotal,
    discountsTotal,
    truckMinimum,
    finalTotal,
    appliedTier,
  } = calculation;

  const usingTruckMinimum = truckMinimum > itemsTotal;

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span>Quote Summary</span>
          <Badge variant="outline">{itemCount} items</Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Items Total</span>
          <span>{formatCurrency(itemsTotal)}</span>
        </div>

        {usingTruckMinimum && (
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">
              Truck Minimum ({appliedTier} load)
            </span>
            <span className="text-primary font-medium">
              {formatCurrency(truckMinimum)}
            </span>
          </div>
        )}

        {accessFactorsTotal > 0 && (
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Access Factors</span>
            <span>+{formatCurrency(accessFactorsTotal)}</span>
          </div>
        )}

        {discountsTotal > 0 && (
          <div className="flex justify-between text-sm text-green-600">
            <span>Discounts</span>
            <span>-{formatCurrency(discountsTotal)}</span>
          </div>
        )}

        <Separator />

        <div className="flex justify-between text-lg font-semibold">
          <span>Total</span>
          <span className="text-primary">{formatCurrency(finalTotal)}</span>
        </div>
      </CardContent>
    </Card>
  );
}
```

### AI Confidence Indicator

```tsx
// components/ai/ConfidenceIndicator.tsx
import { cn } from '@/lib/utils';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger 
} from '@/components/ui/tooltip';
import { CheckCircle, AlertCircle, HelpCircle } from 'lucide-react';

interface ConfidenceIndicatorProps {
  confidence: number; // 0-1
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function ConfidenceIndicator({ 
  confidence, 
  showLabel = false,
  size = 'md' 
}: ConfidenceIndicatorProps) {
  const percentage = Math.round(confidence * 100);
  
  const getConfidenceLevel = () => {
    if (confidence >= 0.85) return { level: 'high', color: 'text-green-500', label: 'High confidence' };
    if (confidence >= 0.6) return { level: 'medium', color: 'text-yellow-500', label: 'Medium confidence' };
    return { level: 'low', color: 'text-red-500', label: 'Low confidence - review recommended' };
  };

  const { level, color, label } = getConfidenceLevel();

  const Icon = level === 'high' 
    ? CheckCircle 
    : level === 'medium' 
      ? HelpCircle 
      : AlertCircle;

  const sizeClasses = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5',
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className={cn('flex items-center gap-1.5', color)}>
          <Icon className={sizeClasses[size]} />
          {showLabel && (
            <span className="text-xs font-medium">{percentage}%</span>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent>
        <p>{label} ({percentage}%)</p>
      </TooltipContent>
    </Tooltip>
  );
}
```

---

## Supabase Edge Functions

### Photo Analysis Edge Function

```typescript
// supabase/functions/analyze-photo/index.ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface DetectedItem {
  name: string;
  category: string;
  confidence: number;
  boundingBox?: { x: number; y: number; width: number; height: number };
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { imageBase64, companyId } = await req.json();

    // Validate input
    if (!imageBase64 || !companyId) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Check credit balance
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    );

    const { data: subscription } = await supabase
      .from('company_subscriptions')
      .select('vision_credits_used, vision_credits_limit')
      .eq('company_id', companyId)
      .single();

    if (!subscription || subscription.vision_credits_used >= subscription.vision_credits_limit) {
      return new Response(
        JSON.stringify({ error: 'Insufficient vision credits' }),
        { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Call AI vision API (primary: GPT-4o, fallback: Gemini)
    let detectedItems: DetectedItem[];
    
    try {
      detectedItems = await analyzeWithOpenAI(imageBase64);
    } catch (primaryError) {
      console.warn('Primary AI failed, using fallback:', primaryError);
      detectedItems = await analyzeWithGemini(imageBase64);
    }

    // Consume credit
    await supabase.rpc('consume_credits', {
      p_company_id: companyId,
      p_credit_type: 'vision',
      p_amount: 1,
    });

    // Log the analysis
    await supabase.from('credit_transactions').insert({
      company_id: companyId,
      credit_type: 'vision',
      amount: -1,
      kind: 'consumption',
      feature_used: 'photo_analysis',
      context: { itemCount: detectedItems.length },
    });

    return new Response(
      JSON.stringify({ 
        items: detectedItems,
        creditsRemaining: subscription.vision_credits_limit - subscription.vision_credits_used - 1,
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Analysis error:', error);
    return new Response(
      JSON.stringify({ error: 'Analysis failed' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});

async function analyzeWithOpenAI(imageBase64: string): Promise<DetectedItem[]> {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${Deno.env.get('OPENAI_API_KEY')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `You are a junk removal pricing expert. Analyze photos and identify items for removal.
Return a JSON array of detected items with: name, category, confidence (0-1).
Categories: furniture, appliance, electronics, debris, other.`,
        },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Identify all items in this photo for junk removal pricing:' },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
          ],
        },
      ],
      response_format: { type: 'json_object' },
    }),
  });

  const data = await response.json();
  return JSON.parse(data.choices[0].message.content).items;
}

async function analyzeWithGemini(imageBase64: string): Promise<DetectedItem[]> {
  // Fallback implementation using Gemini API
  // ... similar structure with Gemini-specific formatting
  return [];
}
```

---

## Database Patterns (RLS)

### Multi-Tenant Row Level Security

```sql
-- migrations/001_quotes_rls.sql

-- Enable RLS on quotes table
ALTER TABLE public.quotes ENABLE ROW LEVEL SECURITY;

-- Helper function to get user's company (avoids recursion)
CREATE OR REPLACE FUNCTION public.get_user_company_id(user_id UUID)
RETURNS UUID
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT company_id 
  FROM public.company_memberships 
  WHERE user_id = $1 
  LIMIT 1;
$$;

-- Policy: Users can only see quotes from their company
CREATE POLICY "quotes_select_policy" ON public.quotes
FOR SELECT USING (
  company_id = public.get_user_company_id(auth.uid())
);

-- Policy: Users can insert quotes for their company
CREATE POLICY "quotes_insert_policy" ON public.quotes
FOR INSERT WITH CHECK (
  company_id = public.get_user_company_id(auth.uid())
);

-- Policy: Users can update their company's quotes
CREATE POLICY "quotes_update_policy" ON public.quotes
FOR UPDATE USING (
  company_id = public.get_user_company_id(auth.uid())
);

-- Policy: Only admins can delete quotes
CREATE POLICY "quotes_delete_policy" ON public.quotes
FOR DELETE USING (
  company_id = public.get_user_company_id(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.company_memberships
    WHERE user_id = auth.uid()
    AND role = 'admin'
  )
);
```

### Credit Consumption Function

```sql
-- migrations/002_credit_functions.sql

CREATE OR REPLACE FUNCTION public.consume_credits(
  p_company_id UUID,
  p_credit_type TEXT,
  p_amount INTEGER
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_used INTEGER;
  v_limit INTEGER;
  v_column_used TEXT;
  v_column_limit TEXT;
BEGIN
  -- Determine column names based on credit type
  v_column_used := p_credit_type || '_credits_used';
  v_column_limit := p_credit_type || '_credits_limit';
  
  IF p_credit_type = 'voice' THEN
    v_column_used := 'voice_minutes_used';
    v_column_limit := 'voice_minutes_limit';
  END IF;

  -- Lock row and check balance
  EXECUTE format(
    'SELECT %I, %I FROM company_subscriptions WHERE company_id = $1 FOR UPDATE',
    v_column_used, v_column_limit
  ) INTO v_used, v_limit USING p_company_id;

  IF v_used + p_amount > v_limit THEN
    RAISE EXCEPTION 'Insufficient credits';
  END IF;

  -- Consume credits
  EXECUTE format(
    'UPDATE company_subscriptions SET %I = %I + $1 WHERE company_id = $2',
    v_column_used, v_column_used
  ) USING p_amount, p_company_id;

  RETURN TRUE;
END;
$$;
```

---

## AI Integration

### Prompt Engineering Pattern

```typescript
// lib/ai/prompts.ts

export const PHOTO_ANALYSIS_PROMPT = `You are PriceMaster AI, an expert junk removal pricing assistant.

TASK: Analyze the provided photo and identify all items suitable for junk removal.

OUTPUT FORMAT:
Return a JSON object with an "items" array. Each item should have:
- name: Specific item name (e.g., "3-seater leather sofa", not just "couch")
- category: One of [furniture, appliance, electronics, debris, other]
- confidence: Your confidence level (0.0 to 1.0)
- estimatedVolume: Cubic yards estimate (optional)
- notes: Any special handling requirements (optional)

GUIDELINES:
1. Be specific with item names - include size, material, condition if visible
2. Group similar small items (e.g., "Box of books" instead of individual books)
3. Flag hazardous items (paint, chemicals, batteries) in notes
4. Estimate volume based on standard truck capacity (15 cubic yards = full load)
5. Lower confidence for partially visible or ambiguous items

EXAMPLE OUTPUT:
{
  "items": [
    {
      "name": "Queen mattress with box spring",
      "category": "furniture",
      "confidence": 0.95,
      "estimatedVolume": 0.8,
      "notes": "Standard size, appears clean"
    }
  ]
}`;

export const MARKET_ANALYSIS_PROMPT = `You are a junk removal market analyst.

Given the following pricing data and market context, provide recommendations:

ANALYSIS REQUIRED:
1. Compare prices against regional averages
2. Identify pricing gaps (items priced too high or low)
3. Suggest new items to add based on market demand
4. Recommend seasonal adjustments

OUTPUT: JSON with recommendations, confidence scores, and reasoning.`;
```

### Multi-Model Fallback Pattern

```typescript
// lib/ai/multi-model.ts

interface AIProvider {
  name: string;
  analyze: (input: string) => Promise<AnalysisResult>;
  priority: number;
}

const providers: AIProvider[] = [
  { name: 'openai-gpt4o', analyze: analyzeWithGPT4o, priority: 1 },
  { name: 'google-gemini', analyze: analyzeWithGemini, priority: 2 },
];

export async function analyzeWithFallback(
  input: string,
  maxRetries = 2
): Promise<AnalysisResult> {
  const sortedProviders = [...providers].sort((a, b) => a.priority - b.priority);
  
  let lastError: Error | null = null;

  for (const provider of sortedProviders) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        console.log(`Attempting ${provider.name} (attempt ${attempt + 1})`);
        
        const result = await provider.analyze(input);
        
        // Log successful provider for analytics
        await logProviderUsage(provider.name, 'success');
        
        return result;
      } catch (error) {
        lastError = error as Error;
        console.warn(`${provider.name} failed:`, error);
        
        await logProviderUsage(provider.name, 'failure', error);
        
        // Exponential backoff before retry
        if (attempt < maxRetries - 1) {
          await sleep(Math.pow(2, attempt) * 1000);
        }
      }
    }
  }

  throw new Error(`All AI providers failed. Last error: ${lastError?.message}`);
}
```

---

## Testing Patterns

### Hook Testing Example

```typescript
// __tests__/hooks/useQuoteCalculation.test.ts
import { renderHook } from '@testing-library/react';
import { useQuoteCalculation } from '@/hooks/useQuoteCalculation';

describe('useQuoteCalculation', () => {
  const mockTruckPricing = {
    '1/8': 150,
    '1/4': 250,
    '1/2': 400,
    'full': 700,
  };

  it('calculates items total correctly', () => {
    const items = [
      { id: '1', basePrice: 50, quantity: 2, volumeCubicYards: 1 },
      { id: '2', basePrice: 100, quantity: 1, volumeCubicYards: 2 },
    ];

    const { result } = renderHook(() =>
      useQuoteCalculation(items, [], [], mockTruckPricing)
    );

    expect(result.current.itemsTotal).toBe(200);
  });

  it('applies truck minimum when items total is lower', () => {
    const items = [
      { id: '1', basePrice: 25, quantity: 1, volumeCubicYards: 3 },
    ];

    const { result } = renderHook(() =>
      useQuoteCalculation(items, [], [], mockTruckPricing)
    );

    expect(result.current.finalTotal).toBe(250); // 1/4 load minimum
    expect(result.current.appliedTier).toBe('1/4');
  });

  it('applies percentage discounts correctly', () => {
    const items = [
      { id: '1', basePrice: 500, quantity: 1, volumeCubicYards: 8 },
    ];
    const discounts = [
      { id: 'd1', type: 'percentage', value: 10 },
    ];

    const { result } = renderHook(() =>
      useQuoteCalculation(items, [], discounts, mockTruckPricing)
    );

    expect(result.current.finalTotal).toBe(450); // 10% off $500
  });
});
```

---

*These samples demonstrate the coding patterns, architecture decisions, and best practices used throughout FlowJunk. All sensitive business logic, API keys, and customer data have been redacted.*
