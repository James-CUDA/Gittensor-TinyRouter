"""The coordinator: ~0.6B encoder + ~10K-parameter head + decision policy.

Modules (filled in against docs/SPEC.md §3):
  encoder.py        -- placeholder; real encoder is slm.CoordinatorEncoder
  slm.py            -- Qwen3-0.6B encode (penultimate) + encode_sequence (full H)
  head.py           -- LinearHead (model×role); CMA-ES θ for submission routing
  triage_head.py    -- TriageHead5Domain / TriageHead20Domain (M1)
  attention_pool.py -- AttentivePool + AttentiveTriageRouter (encode→attn→head)
  policy.py         -- map LinearHead outputs -> (model, role, stop) decision
"""
