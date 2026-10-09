---
updated: 2026-10-09
---
# خارطة الطريق
| المرحلة | الهدف | معيار الإنجاز (قابل للقياس) | الحالة |
|---|---|---|---|
| M1 | نموذج أولي + تدقيق عادل | audit.py: CORTEX > RAG على "all" و"updated" عبر 5 بذور | ✅ (94.4 vs 83.1؛ 66 vs 0.2) |
| M2 | embeddings حقيقية | استبدال الهاش بـ sentence-transformers؛ لا تراجع في all، multihop ≥ 30% | ⬜ |
| M3 | LLM داخل الحلقة | دقة الجواب النهائي (لا الاسترجاع) ≥ RAG+5 نقاط على hold-out | ⬜ |
| M4 | معيار عام | LongMemEval: تقرير مقارنة مع RAG و HippoRAG | ⬜ |
| M5 | ربط بـ agent-brain | أمر `brain.py recall "<q>"` يعيد ≤ 20 سطراً | ⬜ |
## Now / Next / Later
- Now: M2 · Next: M3, M5 · Later: M4، طبقة القشرة (Cartridges/LoRA على GPU)
## خارج النطاق صراحةً
- تدريب نماذج من الصفر.
