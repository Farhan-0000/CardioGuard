# MedBridge v3 — Master Project Context

> **Document Type:** Authoritative Technical Specification
> **Status:** Consolidated from 6 architecture artifacts + ADL review
> **Date:** 2026-08-15
> **Conflict Resolution:** Where source artifacts contradict each other, this document states the **resolved position** and cites the originating conflict by ADL reference.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Functional Requirements](#2-functional-requirements)
3. [Non-Functional Requirements](#3-non-functional-requirements)
4. [System Boundaries](#4-system-boundaries)
5. [User Roles](#5-user-roles)
6. [Core Workflows](#6-core-workflows)
7. [Components and Responsibilities](#7-components-and-responsibilities)
8. [Data Entities](#8-data-entities)
9. [AI Pipeline Description](#9-ai-pipeline-description)
10. [Storage Architecture](#10-storage-architecture)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Assumptions](#12-assumptions)
13. [Constraints](#13-constraints)
14. [Risks](#14-risks)

---

## 1. Project Purpose

### 1.1 Problem Statement

Hypertension patients seeking clinical guidance online face two categories of harm:

1. **Ungrounded medical advice** — generative AI models fabricate drug interactions, dosages, or recommendations not supported by authoritative clinical guidelines.
2. **Unsafe personalization** — systems provide patient-specific guidance without adequate medical context (e.g., recommending a beta-blocker without knowing the patient has asthma).

Standard Retrieval-Augmented Generation (RAG) systems retrieve evidence but do not evaluate whether sufficient patient context or guideline evidence exists before generating a response. This creates a safety gap.

### 1.2 Project Definition

MedBridge is an **AI-assisted clinical communication system** for hypertension patients. It uses a **Two-Stage Answerability Engine** to evaluate both patient context sufficiency and retrieved evidence adequacy before generating any clinical guidance.

The system does **not** diagnose, prescribe, or replace a physician. It provides evidence-grounded informational guidance with citations to authoritative clinical guidelines, and deterministically refuses or escalates when safety conditions are not met.

### 1.3 Core Safety Differentiator

MedBridge introduces two sequential decision gates that standard RAG systems lack:

| Gate | Name | Position | Purpose |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Context Gate | Pre-retrieval | Prevents personalized guidance when patient context is insufficient |
| **Gate 2** | Evidence Gate | Post-retrieval | Prevents generation when retrieved evidence is insufficient or query is dangerous |

These gates enforce **deterministic safety routing** into exactly one of five actions:

| Action | Definition |
| :--- | :--- |
| `SOFT-ASK` | Request missing patient context via a clarifying question (pre-retrieval short-circuit) |
| `ANSWER` | Deliver personalized, evidence-grounded clinical guidance with inline citations |
| `GENERALIZE` | Deliver unpersonalized general health information when specific evidence is insufficient |
| `ABSTAIN` | Safely refuse to answer when no relevant evidence exists or the query is out of domain |
| `ESCALATE` | Refer to a healthcare provider when the query indicates clinical danger or emergency |

> [!NOTE]
> **Conflict Resolved (ADL-008):** The High-Level Architecture diagram uses legacy terms ("Safe Answer", "Refusal", "Ask"). This document standardizes on the 5-action enum above. All implementation must use these canonical names exclusively.

---

## 2. Functional Requirements

### 2.1 Patient Interaction

| ID | Requirement | Source |
| :--- | :--- | :--- |
| FR-01 | The system shall accept free-text clinical questions from hypertension patients via a web chat interface | Diagram 1, Spec |
| FR-02 | The system shall maintain conversational context across multiple turns within a session | Diagram 4 |
| FR-03 | The system shall display the routing action taken (`ANSWER`, `SOFT-ASK`, `GENERALIZE`, `ABSTAIN`, `ESCALATE`) alongside each response | Diagram 2 |
| FR-04 | The system shall display citation excerpts with source attribution (guideline name, section) for every generated response | Diagrams 3, 4 |
| FR-05 | The system shall request missing patient context via `SOFT-ASK` for a maximum of **2 consecutive turns** per session before falling back to `GENERALIZE` | Diagram 4, ADL-013 |

### 2.2 Clinical Safety

| ID | Requirement | Source |
| :--- | :--- | :--- |
| FR-06 | The system shall **never** generate clinical guidance without first evaluating patient context sufficiency (Gate 1) | Diagrams 2, 3 |
| FR-07 | The system shall **never** generate personalized guidance without first evaluating retrieved evidence sufficiency (Gate 2) | Diagrams 2, 3 |
| FR-08 | The system shall route to `ESCALATE` with a pre-vetted response template when emergency symptoms are detected (e.g., BP > 180/120 with organ damage symptoms) | Diagram 3, ADL-002 |
| FR-09 | For `ABSTAIN` and `ESCALATE` actions, the system shall return deterministic, pre-approved response templates and shall **not** route through LLM generation | ADL-015 |
| FR-10 | The system shall log every gate decision (action, rationale, evidence scores) to an immutable audit trail | Diagram 2 |

### 2.3 Knowledge Management

| ID | Requirement | Source |
| :--- | :--- | :--- |
| FR-11 | The system shall retrieve evidence from three curated clinical knowledge sources: AHA/ACC 2025, ESC/ESH 2024, and MedlinePlus | Diagrams 2, 3, 5 |
| FR-12 | The system shall use hybrid retrieval (BM25 sparse + dense vector search) with Reciprocal Rank Fusion (RRF) to produce Top-20 candidate chunks | Diagrams 2, 3 |
| FR-13 | The system shall rerank Top-20 candidates to Top-5 evidence chunks using a cross-encoder model before passing to the Evidence Gate | Diagrams 2, 3, 4 |
| FR-14 | The system shall support offline one-time ingestion of clinical guideline PDFs into the vector database | Diagram 5 |

### 2.4 Session & State Management

| ID | Requirement | Source |
| :--- | :--- | :--- |
| FR-15 | The system shall use an append-only event log to record all patient context facts extracted during a session | Diagram 2 |
| FR-16 | The system shall derive a materialized context snapshot (JSONB) from the event log via a deterministic state projector | ADL-001 |
| FR-17 | The system shall load the existing session snapshot and `soft_ask_count` from the database at the start of each request | ADL-014 |
| FR-18 | The system shall track `soft_ask_count` per session and enforce the loop-breaker threshold (max 2) | Diagram 4, ADL-013 |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
| :--- | :--- | :--- |
| NFR-01 | End-to-end response latency (PROCEED path, 4 LLM calls + retrieval) | ≤ 8 seconds |
| NFR-02 | SOFT-ASK short-circuit latency (2 LLM calls, no retrieval) | ≤ 3 seconds |
| NFR-03 | Emergency fast-path latency (deterministic classifier, no LLM) | ≤ 500 ms |
| NFR-04 | Cross-encoder reranking (20 pairs on CPU) | ≤ 1.5 seconds |
| NFR-05 | Concurrent user sessions (semester deployment) | ≥ 5 simultaneous |

### 3.2 Reliability

| ID | Requirement | Target |
| :--- | :--- | :--- |
| NFR-06 | LLM structured output parse success rate after retries | ≥ 95% |
| NFR-07 | Graceful degradation on permanent LLM parse failure | Route to `GENERALIZE` or `ESCALATE`, never HTTP 500 |
| NFR-08 | Groq API availability fallback | Retry with exponential backoff (3 attempts, 1s/2s/4s) |

### 3.3 Safety & Compliance

| ID | Requirement | Target |
| :--- | :--- | :--- |
| NFR-09 | Gate decision reproducibility for identical inputs | Deterministic (temperature 0.0, seed 42) |
| NFR-10 | Audit log completeness | 100% of requests logged with gate decisions and rationale |
| NFR-11 | PII handling acknowledgment | Documented as architectural gap; anonymization layer deferred to future phase |

### 3.4 Evaluation

| ID | Requirement | Target |
| :--- | :--- | :--- |
| NFR-12 | MedBridge-AQ benchmark coverage | ≥ 200 annotated conversational vignettes |
| NFR-13 | Inter-annotator agreement (Cohen's κ) | ≥ 0.6 |
| NFR-14 | RABBITS adversarial evaluation | RxNorm brand ↔ generic substitution test suite |
| NFR-15 | Baseline comparisons | Vanilla RAG, Prompt-Only Safety, Direct LLM, Gate Ablations |

---

## 4. System Boundaries

### 4.1 In Scope

```
┌─────────────────────────────────────────────────────────────────┐
│                    MedBridge System Boundary                     │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────────┐  │
│  │ Chat UI  │→ │ FastAPI  │→ │  Answerability Engine          │  │
│  │ (Web)    │  │ Backend  │  │  (Extractor, Gates, Generator) │  │
│  └──────────┘  └──────────┘  └───────────────────────────────┘  │
│                     │                      │                     │
│              ┌──────┴──────┐        ┌──────┴──────┐             │
│              │ PostgreSQL  │        │   Qdrant    │             │
│              │ (Sessions)  │        │  (Vectors)  │             │
│              └─────────────┘        └─────────────┘             │
│                                                                  │
│  ┌─────────────────────┐    ┌────────────────────────────────┐  │
│  │ Ingestion Pipeline  │    │ Deterministic State Projector  │  │
│  │ (Offline CLI)       │    │ (Event → Snapshot Reducer)     │  │
│  └─────────────────────┘    └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  External (Cloud)  │
                    │  Groq / OpenAI API │
                    └───────────────────┘
```

### 4.2 Out of Scope

| Exclusion | Rationale |
| :--- | :--- |
| User authentication and authorization | Academic prototype; no login system |
| Multi-tenancy or role-based access control | Single-user sessions only |
| Real-time vital sign integration (EHR/FHIR) | No live clinical data feeds |
| Drug interaction databases (RxNorm live queries) | Offline guideline text only |
| Mobile native applications | Web browser only |
| PII anonymization enforcement | Acknowledged gap (ADL-017); deferred |
| Production-grade horizontal scaling | Single-node deployment for semester |
| Multilingual support | English only |

---

## 5. User Roles

| Role | Description | Capabilities |
| :--- | :--- | :--- |
| **Hypertension Patient** | Primary end user. An adult diagnosed with or managing hypertension who seeks informational guidance about their condition, medications, lifestyle, or blood pressure readings. | Submit free-text clinical questions; receive grounded guidance with citations; view conversation history within a session. |
| **System Administrator** (implicit) | The developer or researcher deploying and maintaining the system. | Run ingestion pipeline; manage Docker containers; review audit logs; configure environment variables and API keys. |
| **Evaluator** (implicit) | A researcher running the MedBridge-AQ benchmark or adversarial tests. | Execute evaluation scripts against the API; compare routing accuracy across baselines. |

> [!NOTE]
> **Assumption (ASM-01):** No formal authentication exists. Sessions are identified by a `session_id` (UUID) created on first contact. Any user with the session URL can resume a session. This is acceptable for an academic prototype but would be a security defect in production.

---

## 6. Core Workflows

### 6.1 Primary Request Flow (PROCEED Path)

This is the full 8-stage pipeline when patient context is sufficient and the system proceeds to retrieval.

```mermaid
sequenceDiagram
    participant P as Patient
    participant UI as Chat UI
    participant API as FastAPI Backend
    participant DB as PostgreSQL
    participant CE as Context Extractor<br/>(LLM Call 1)
    participant SP as State Projector
    participant CG as Context Gate<br/>(LLM Call 2)
    participant HR as Hybrid Retriever
    participant QD as Qdrant
    participant RR as Cross-Encoder Reranker
    participant EG as Evidence Gate<br/>(LLM Call 3)
    participant RG as Response Generator<br/>(LLM Call 4)

    P->>UI: 1. Type clinical message
    UI->>API: 2. POST /api/sessions/{id}/messages
    API->>DB: 2.5 Load snapshot + soft_ask_count
    DB-->>API: Existing session state
    API->>CE: 3. User message + snapshot
    CE-->>API: 4. ExtractorOutput JSON (delta events + search_query)
    API->>SP: 5. Apply delta events to event log
    SP->>DB: 6. Append events + write updated snapshot
    API->>CG: 7. Snapshot + message → Evaluate sufficiency
    CG-->>API: 8. ContextGateOutput {action: PROCEED}
    API->>HR: 9. Reformulated search query
    HR->>QD: 10. BM25 + Dense hybrid search
    QD-->>HR: 11. Top-20 candidate chunks
    HR->>RR: 12. Rerank 20 candidates
    RR-->>API: 13. Top-5 reranked evidence package
    API->>EG: 14. Evidence + snapshot → Evaluate sufficiency
    EG-->>API: 15. EvidenceGateOutput {action: ANSWER}
    API->>RG: 16. Evidence + snapshot + action → Generate
    RG-->>API: 17. Synthesized response + inline citations
    API->>DB: 18. Write audit log
    API-->>UI: 19. HTTP 200 JSON response
    UI-->>P: 20. Render guidance + citations panel
```

### 6.2 SOFT-ASK Short-Circuit (Insufficient Context)

When patient context is insufficient and `soft_ask_count < 2`:

```
Patient message → Context Extractor (LLM 1) → State Projector → Context Gate (LLM 2)
    → [INSUFFICIENT + count < 2]
    → Response Generator (LLM 4, SOFT-ASK template)
    → Increment soft_ask_count → Return clarifying question
```

### 6.3 Loop-Breaker Fallback (Context Still Insufficient After 2 Attempts)

When context remains insufficient but `soft_ask_count ≥ 2`:

```
Patient message → Context Extractor (LLM 1) → State Projector → Context Gate (LLM 2)
    → [INSUFFICIENT + count ≥ 2]
    → Force-route to GENERALIZE
    → Hybrid Retrieval → Reranking → Response Generator (LLM 4, general guidance)
    → Return unpersonalized guidance with MedlinePlus citations
```

> [!IMPORTANT]
> **Conflict Resolved (ADL-013):** The sequence diagram omitted this branch entirely. This document adds it as a mandatory execution path. Without it, the system enters an infinite SOFT-ASK loop.

### 6.4 Emergency Fast-Path

When the input contains critical red-flag patterns:

```
Patient message → API Gateway → Emergency Triage Classifier (deterministic)
    → [EMERGENCY DETECTED]
    → Return pre-vetted ESCALATE template immediately
    → Bypass entire AI pipeline
```

> [!IMPORTANT]
> **New Component (ADL-002):** This fast-path does not exist in the original diagrams. This document adds it as a required component to avoid 3–6 second latency on life-threatening queries.

### 6.5 ABSTAIN / ESCALATE Deterministic Response

When the Evidence Gate routes to `ABSTAIN` or `ESCALATE`:

```
Evidence Gate → {action: ABSTAIN or ESCALATE}
    → Bypass LLM Call 4 (Response Generator)
    → Return pre-vetted template from application configuration
    → Write audit log
```

> [!IMPORTANT]
> **Conflict Resolved (ADL-015):** The sequence diagram routes all actions through LLM Call 4. This document restricts LLM Call 4 to `ANSWER` and `GENERALIZE` only. `ABSTAIN` and `ESCALATE` use deterministic templates.

### 6.6 Offline Ingestion Pipeline

```
Clinical Guideline PDFs (AHA/ACC 2025, ESC/ESH 2024, MedlinePlus)
    → PyMuPDF text extraction
    → Section-aware chunking (with metadata preservation)
    → Dense embedding (bge-small-en-v1.5) + Sparse tokenization (fastembed BM25)
    → Upsert to Qdrant collection "clinical_guidelines"
    → Schema validation
```

---

## 7. Components and Responsibilities

### 7.1 Component Registry

| Layer | Component | Technology | Responsibility |
| :--- | :--- | :--- | :--- |
| **L1 – Presentation** | Patient Chat UI | Streamlit (Python :8501) | Render chat interface, display responses with citations, manage session UX |
| **L2 – API** | FastAPI Backend | FastAPI / Uvicorn (Python :8000) | REST API gateway, session management, orchestrate AI pipeline, serve responses |
| **L2 – API** | Emergency Triage Classifier | Deterministic regex/rules | Pre-pipeline fast-path for emergency detection (ADL-002) |
| **L3 – AI Pipeline** | Context Extractor | LLM Call 1 (structured output) | Extract structured medical facts from raw patient message; produce delta events and reformulated search query |
| **L3 – AI Pipeline** | Deterministic State Projector | Python business logic | Apply typed delta events to the event log; fold event stream into context snapshot JSONB using deterministic rules (ADL-001) |
| **L3 – AI Pipeline** | Context Gate | LLM Call 2 (classification) | Binary classification: evaluate whether the accumulated context snapshot contains sufficient patient information for safe personalized retrieval |
| **L3 – AI Pipeline** | Hybrid Retriever | qdrant-client + fastembed | Execute parallel BM25 sparse + dense vector searches against Qdrant; fuse results via RRF; return Top-20 candidate chunks |
| **L3 – AI Pipeline** | Cross-Encoder Reranker | sentence-transformers | Rerank Top-20 candidates to Top-5 using pairwise query-document scoring |
| **L3 – AI Pipeline** | Evidence Gate | LLM Call 3 (classification) | Evaluate evidence sufficiency; route to ANSWER / GENERALIZE / ABSTAIN / ESCALATE |
| **L3 – AI Pipeline** | Response Generator | LLM Call 4 (constrained generation) | Synthesize patient-facing response with inline citations using XML-isolated prompt; only for ANSWER and GENERALIZE actions |
| **L3 – AI Pipeline** | Resilient LLM Wrapper | Python retry logic | Wrap all LLM calls with 3-attempt exponential backoff, JSON repair, and deterministic fallback on permanent failure (ADL-004) |
| **L4 – Storage** | PostgreSQL | Docker Container (:5432) | Append-only event log, context snapshots, sessions, audit logs |
| **L4 – Storage** | Qdrant | Docker Container (:6333) | Dense + sparse vector storage for clinical guideline chunks |
| **L5 – Knowledge** | Clinical Guideline PDFs | Local filesystem | AHA/ACC 2025, ESC/ESH 2024, MedlinePlus source documents |
| **L5 – Knowledge** | Ingestion Worker | Python CLI script | One-time offline PDF parsing, chunking, embedding, and Qdrant indexing |
| **External** | LLM Inference API | Groq Cloud (Llama 3.1 8B) / OpenAI (GPT-4o-mini) | Remote LLM inference for all 4 pipeline calls via HTTPS REST |

> [!NOTE]
> **Conflict Resolved (ADL-007):** The deployment diagram labels the frontend as "React / Streamlit — Node.js Process (Port 3000)". Streamlit is a Python runtime on port 8501, not Node.js. This document standardizes on **Streamlit** as the semester frontend, as it aligns with the Python-only backend stack and removes the Node.js dependency.

> [!NOTE]
> **Conflict Resolved (ADL-023):** The original diagrams assign query reformulation to Context Gate (LLM Call 2). This document moves query reformulation to the Context Extractor (LLM Call 1), restricting Context Gate to pure binary classification only.

### 7.2 Component Interaction Matrix

| Producer → Consumer | Data Exchanged | Protocol |
| :--- | :--- | :--- |
| Chat UI → FastAPI Backend | `{ message, session_id }` | HTTP REST / JSON |
| FastAPI Backend → PostgreSQL | SQL queries (event append, snapshot read/write, audit write) | SQL via SQLAlchemy (:5432) |
| FastAPI Backend → Qdrant | Hybrid vector search queries | gRPC / REST (:6333) |
| FastAPI Backend → Groq / OpenAI | LLM prompts (structured JSON output) | HTTPS REST (:443) |
| FastAPI Backend → Chat UI | `{ response_text, action, citations[], soft_ask_count }` | HTTP REST / JSON |
| Ingestion Worker → Qdrant | Chunked vectors with payload metadata | gRPC / REST (:6333) |

---

## 8. Data Entities

### 8.1 Session

| Field | Type | Description |
| :--- | :--- | :--- |
| `session_id` | UUID (PK) | Unique identifier for a patient conversation session |
| `created_at` | TIMESTAMP | Session creation time |
| `last_active_at` | TIMESTAMP | Last message timestamp |

### 8.2 Clinical Event (Append-Only Log)

| Field | Type | Description |
| :--- | :--- | :--- |
| `event_id` | UUID (PK) | Unique event identifier |
| `session_id` | UUID (FK → sessions) | Parent session |
| `event_type` | ENUM | One of: `BP_READING`, `MEDICATION_ADDED`, `MEDICATION_STOPPED`, `SYMPTOM_REPORTED`, `DEMOGRAPHIC`, `LAB_RESULT` |
| `payload` | JSONB | Event-specific structured data |
| `created_at` | TIMESTAMP | Event creation time |

**Event Payload Examples:**

```json
// BP_READING
{ "systolic": 145, "diastolic": 92, "context": "morning reading" }

// MEDICATION_ADDED
{ "drug_name": "Amlodipine", "dosage": "5mg", "frequency": "daily" }

// MEDICATION_STOPPED
{ "drug_name": "Lisinopril", "reason": "patient reported cough" }

// DEMOGRAPHIC
{ "age": 58, "sex": "male", "smoking_status": "former" }
```

### 8.3 Context Snapshot (Materialized View)

| Field | Type | Description |
| :--- | :--- | :--- |
| `session_id` | UUID (FK → sessions, UNIQUE) | One snapshot per session |
| `snapshot` | JSONB | Accumulated patient context |
| `soft_ask_count` | INTEGER | Number of consecutive SOFT-ASK responses issued |
| `updated_at` | TIMESTAMP | Last projection time |

**Snapshot JSONB Structure:**

```json
{
  "demographics": { "age": 58, "sex": "male", "smoking": "former" },
  "current_medications": [
    { "drug": "Amlodipine", "dosage": "5mg", "frequency": "daily" }
  ],
  "discontinued_medications": [
    { "drug": "Lisinopril", "reason": "cough", "stopped_at": "..." }
  ],
  "recent_bp_readings": [
    { "systolic": 145, "diastolic": 92, "context": "morning" }
  ],
  "symptoms": ["ankle swelling", "occasional headache"],
  "comorbidities": ["type 2 diabetes"],
  "lab_results": []
}
```

> [!IMPORTANT]
> **Conflict Resolved (ADL-001):** This snapshot is produced by the **Deterministic State Projector**, not by the LLM. The LLM (Context Extractor) produces typed delta events. The Projector applies business rules (e.g., a `MEDICATION_STOPPED` event removes the drug from `current_medications` and adds it to `discontinued_medications`). This guarantees idempotent replay.

### 8.4 Audit Log

| Field | Type | Description |
| :--- | :--- | :--- |
| `audit_id` | UUID (PK) | Unique log entry |
| `session_id` | UUID (FK → sessions) | Parent session |
| `request_message` | TEXT | Original patient message |
| `gate_1_action` | ENUM | `SOFT-ASK` or `PROCEED` |
| `gate_1_rationale` | TEXT | Context Gate reasoning |
| `gate_2_action` | ENUM (nullable) | `ANSWER`, `GENERALIZE`, `ABSTAIN`, `ESCALATE` (null if short-circuited) |
| `gate_2_rationale` | TEXT (nullable) | Evidence Gate reasoning |
| `final_action` | ENUM | Resolved action delivered to patient |
| `evidence_chunk_ids` | TEXT[] | Chunk IDs used in response generation |
| `created_at` | TIMESTAMP | Log timestamp |

### 8.5 Qdrant Vector Collection

| Property | Value |
| :--- | :--- |
| **Collection Name** | `clinical_guidelines` |
| **Dense Vector** | name=`dense`, dimensions=384, distance=Cosine, model=`bge-small-en-v1.5` |
| **Sparse Vector** | name=`sparse`, model=`Qdrant/bm25` (via fastembed client-side generation) |

**Payload Schema per Point:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | string | Unique identifier for the chunk |
| `guideline_id` | string | `AHA_ACC_2025`, `ESC_ESH_2024`, or `MEDLINEPLUS` |
| `section_title` | string | Section/subsection header from the source document |
| `page_number` | integer | Source PDF page number |
| `chunk_text` | string | Full text content of the chunk |
| `source_url` | string | URL or document reference for citation |

> [!NOTE]
> **Conflict Resolved (ADL-022):** Qdrant does not compute BM25 sparse vectors server-side. The `fastembed` library with model `Qdrant/bm25` must be used client-side at both ingestion time and query time to produce sparse vectors. This is a mandatory dependency.

---

## 9. AI Pipeline Description

### 9.1 Pipeline Overview

The AI pipeline consists of **4 LLM inference calls** to a shared reasoning engine, plus **1 deterministic retrieval-reranking stage** and **1 deterministic state projection step**. All LLM calls use a single model endpoint (Groq Llama 3.1 8B or GPT-4o-mini).

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Input   │───▶│ Context  │───▶│  State  │───▶│ Context  │───▶│ Hybrid   │───▶│ Evidence │───▶│ Response │───▶│  Output  │
│  (Raw    │    │Extractor │    │Projector│    │  Gate    │    │Retrieval │    │  Gate    │    │Generator │    │ (Safe    │
│  Message)│    │ LLM #1   │    │(Determ.)│    │ LLM #2  │    │+ Rerank  │    │ LLM #3  │    │ LLM #4   │    │Response) │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                  │                                              │
                                                  ▼                                              ▼
                                           [SOFT-ASK]                                [ABSTAIN/ESCALATE]
                                          short-circuit                             deterministic template
                                          (if count < 2)                              (bypass LLM #4)
                                                  │
                                                  ▼
                                         [count ≥ 2]
                                        force GENERALIZE
                                        (proceed to retrieval)
```

### 9.2 LLM Call 1 — Context Extractor

| Property | Value |
| :--- | :--- |
| **Purpose** | Extract structured medical facts from the raw patient message; produce typed delta events for the event log; generate a reformulated clinical search query |
| **Input** | Raw user message + current context snapshot (JSONB) |
| **Output Schema** | `ExtractorOutput` |
| **Temperature** | 0.0 |

**`ExtractorOutput` Schema:**

```
{
  "delta_events": [
    {
      "event_type": "BP_READING | MEDICATION_ADDED | MEDICATION_STOPPED | SYMPTOM_REPORTED | DEMOGRAPHIC | LAB_RESULT",
      "payload": { ... }
    }
  ],
  "search_query": "reformulated clinical search query string",
  "raw_intent": "brief natural-language summary of what the patient is asking"
}
```

> [!NOTE]
> **Conflict Resolved (ADL-023):** Query reformulation (`search_query`) was originally assigned to LLM Call 2 (Context Gate). This document moves it to LLM Call 1, as the Extractor already performs structured analysis and can generate the search query as a natural byproduct without overloading the safety-critical Context Gate.

### 9.3 LLM Call 2 — Context Gate (Gate 1, Pre-Retrieval)

| Property | Value |
| :--- | :--- |
| **Purpose** | Pure binary classification: determine whether the accumulated context snapshot contains sufficient patient information for safe personalized retrieval |
| **Input** | Context snapshot (JSONB) + raw user message + `raw_intent` from Extractor |
| **Output Schema** | `ContextGateOutput` |
| **Temperature** | 0.0, seed 42 |
| **Decision Rule** | If essential clinical parameters needed for personalization are missing → `SOFT-ASK`. Otherwise → `PROCEED`. |

**`ContextGateOutput` Schema:**

```
{
  "action": "SOFT-ASK | PROCEED",
  "missing_fields": ["age", "current_medications"],
  "rationale": "Patient has not provided age or current medication list, which are required for personalized BP target guidance."
}
```

**State Machine Logic (post-gate):**

| Condition | Result |
| :--- | :--- |
| `action == PROCEED` | Continue to Hybrid Retrieval (Stage 4) |
| `action == SOFT-ASK` AND `soft_ask_count < 2` | Short-circuit to Response Generator with SOFT-ASK template; increment `soft_ask_count` |
| `action == SOFT-ASK` AND `soft_ask_count ≥ 2` | Force override to `GENERALIZE`; proceed to Hybrid Retrieval with generic query |

### 9.4 Hybrid Retrieval & Reranking (Stages 4–5)

| Stage | Component | Details |
| :--- | :--- | :--- |
| **4a. Sparse Search** | BM25 via fastembed | Client-side tokenization of `search_query` → sparse vector → Qdrant sparse search |
| **4b. Dense Search** | bge-small-en-v1.5 | Encode `search_query` → 384-dim dense vector → Qdrant dense search |
| **4c. Fusion** | Reciprocal Rank Fusion (RRF) | Merge sparse + dense result sets; produce unified Top-20 ranked candidates |
| **5. Reranking** | Cross-Encoder | Score all 20 query-document pairs; return Top-5 with reranker scores |

**Reranker Model Selection (Resolved — ADL-020):**

The deployment runs on CPU without GPU. Two acceptable options:

| Option | Model | Parameters | CPU Latency (20 pairs) |
| :--- | :--- | :---: | :--- |
| **A (Preferred)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ~100–200ms |
| B | `bge-reranker-v2-m3` (ONNX quantized) | 560M | ~500–800ms |

The chosen reranker **must** be offloaded to a background thread via `asyncio.to_thread()` to avoid blocking the FastAPI event loop.

### 9.5 LLM Call 3 — Evidence Gate (Gate 2, Post-Retrieval)

| Property | Value |
| :--- | :--- |
| **Purpose** | Evaluate whether the Top-5 retrieved evidence chunks are sufficient to safely answer the patient's query |
| **Input** | Top-5 evidence chunks (text + metadata) + context snapshot + user query |
| **Output Schema** | `EvidenceGateOutput` |
| **Temperature** | 0.0, seed 42 |

**`EvidenceGateOutput` Schema:**

```
{
  "action": "ANSWER | GENERALIZE | ABSTAIN | ESCALATE",
  "evidence_sufficient": true,
  "rationale": "Top-ranked chunks directly address Amlodipine dosing for a 58-year-old male with diabetes per AHA/ACC 2025 Section 8.2."
}
```

**Decision Matrix (Resolved — ADL-024):**

| Action | Trigger Criteria |
| :--- | :--- |
| **ANSWER** | Top-ranked evidence directly addresses the patient's specific clinical parameters (drug, dosage, comorbidity combination) with high semantic relevance |
| **GENERALIZE** | Evidence covers the general disease topic (hypertension management) but lacks specificity for the patient's exact scenario (e.g., no match for their specific drug combination or comorbidity) |
| **ABSTAIN** | Top-ranked evidence has low relevance, or the query falls outside the cardiovascular knowledge base domain (e.g., pediatric dosing, oncology, orthopedics) |
| **ESCALATE** | Query indicates acute clinical danger: symptoms consistent with hypertensive emergency (BP > 180/120 mmHg with end-organ damage), stroke, or acute coronary syndrome, regardless of evidence availability |

### 9.6 LLM Call 4 — Response Generator

| Property | Value |
| :--- | :--- |
| **Purpose** | Synthesize a patient-facing response grounded in retrieved evidence, using XML-isolated prompt structure to prevent fact-evidence contamination |
| **Input** | Action decision + evidence package + context snapshot + user query |
| **Output Schema** | `ResponseGeneratorOutput` |
| **Temperature** | 0.3 (for natural language fluency) |
| **Invoked For** | `ANSWER` and `GENERALIZE` actions **only** |
| **Not Invoked For** | `ABSTAIN` and `ESCALATE` (deterministic templates used instead) |

**XML-Isolated Prompt Structure (Resolved — ADL-025):**

```xml
<system_instructions>
  You are a clinical communication assistant. Generate a response using
  ONLY the evidence provided in <clinical_evidence>. Do not assume or
  infer patient information not present in <patient_context>. Follow
  the routing action: {action_decision}.
  Include inline citation markers [1], [2], etc. for each clinical claim.
</system_instructions>

<patient_context>
  {context_snapshot_json}
</patient_context>

<clinical_evidence>
  <chunk id="1" source="{guideline_id}" section="{section_title}">
    {chunk_text}
  </chunk>
  <!-- up to 5 chunks -->
</clinical_evidence>

<user_query>
  {original_user_message}
</user_query>
```

**`ResponseGeneratorOutput` Schema:**

```
{
  "response_text": "Based on current guidelines, ... [1]. Your Amlodipine dosage ... [2].",
  "citations": [
    { "marker": "[1]", "chunk_id": "aha_2025_s8_c3", "source": "AHA/ACC 2025", "section": "Section 8.2", "excerpt": "..." },
    { "marker": "[2]", "chunk_id": "aha_2025_s8_c7", "source": "AHA/ACC 2025", "section": "Section 8.5", "excerpt": "..." }
  ]
}
```

### 9.7 LLM Inference Configuration

| Parameter | LLM Calls 1–3 (Extraction & Gates) | LLM Call 4 (Generation) |
| :--- | :--- | :--- |
| `temperature` | 0.0 | 0.3 |
| `top_p` | 1.0 | 0.95 |
| `seed` | 42 (where supported) | — |
| `max_tokens` | 512 | 1024 |
| `response_format` | JSON mode / structured output | JSON mode / structured output |

> [!NOTE]
> **Gap Resolved (ADL-026):** No original artifact specified these parameters. They are now mandatory configuration for the orchestrator pipeline.

### 9.8 Resilient LLM Wrapper

All 4 LLM calls are wrapped in a resilience layer:

```
Attempt 1 → Parse JSON → Validate Pydantic schema → Success
         ↓ (parse failure)
    JSON repair (fix trailing commas, unclosed brackets)
Attempt 2 → Parse JSON → Validate Pydantic schema → Success
         ↓ (parse failure)
    Re-prompt with explicit schema reminder
Attempt 3 → Parse JSON → Validate Pydantic schema → Success
         ↓ (permanent failure)
    Deterministic fallback → GENERALIZE (Calls 1–3) or ESCALATE (Call 4)
    Log failure to audit trail → Never return HTTP 500 to patient
```

---

## 10. Storage Architecture

### 10.1 PostgreSQL (Relational + Event Store)

| Concern | Implementation |
| :--- | :--- |
| **Pattern** | Event Sourcing with materialized JSONB snapshot |
| **Tables** | `sessions`, `clinical_events`, `context_snapshots`, `audit_logs` |
| **Write Model** | Append-only to `clinical_events`; upsert to `context_snapshots` |
| **Read Model** | `context_snapshots` (single row per session, token-efficient for LLM prompts) |
| **Projection** | Deterministic State Projector folds event stream → snapshot (not the LLM) |
| **ORM** | SQLAlchemy 2.0 + asyncpg (async driver) |

> [!IMPORTANT]
> **Conflict Resolved (ADL-005):** The requirements document stated PostgreSQL does not need a Docker container. The deployment diagram shows it containerized. This document standardizes on **Docker Compose** managing both PostgreSQL (:5432) and Qdrant (:6333) containers, which provides environment isolation and reproducible setup.

### 10.2 Qdrant (Vector Store)

| Concern | Implementation |
| :--- | :--- |
| **Pattern** | Hybrid dense + sparse vector search with RRF fusion |
| **Collection** | `clinical_guidelines` |
| **Dense Model** | `bge-small-en-v1.5` (384 dimensions, Cosine distance) |
| **Sparse Model** | `Qdrant/bm25` via `fastembed` (client-side tokenization) |
| **Ingestion** | One-time offline batch via `IngestionWorker` CLI |
| **Query-Time** | Hybrid search returning Top-20 candidates with payload metadata |

### 10.3 Data Lifecycle

```mermaid
flowchart LR
    subgraph Ingestion ["Offline Ingestion (One-Time)"]
        PDF["Clinical Guideline PDFs"] --> Parse["PyMuPDF Extraction"]
        Parse --> Chunk["Section-Aware Chunking"]
        Chunk --> Embed["Dense + Sparse Embedding"]
        Embed --> QD[("Qdrant Collection")]
    end

    subgraph Runtime ["Runtime (Per Request)"]
        MSG["Patient Message"] --> EXT["Context Extractor"]
        EXT --> EVT["Delta Events"]
        EVT --> PROJ["State Projector"]
        PROJ --> PG_W[("PostgreSQL Write")]
        PG_W --> SNAP["Context Snapshot"]

        SNAP --> GATE["Context Gate"]
        GATE -->|PROCEED| SEARCH["Hybrid Search"]
        SEARCH --> QD
        QD --> RANK["Rerank → Top-5"]
        RANK --> EGATE["Evidence Gate"]
        EGATE --> GEN["Response Generator"]
        GEN --> AUDIT[("Audit Log Write")]
    end
```

---

## 11. Deployment Architecture

### 11.1 Topology

**Single-node local deployment** for the semester scope. All services run on one developer machine.

```
┌─────────────────────────────────────────────────────┐
│          Local Development Machine                   │
│                                                      │
│  ┌──────────────────────┐                           │
│  │  Streamlit Frontend   │                           │
│  │  Python :8501         │                           │
│  └──────────┬───────────┘                           │
│             │ HTTP :8000                             │
│  ┌──────────▼───────────┐      ┌──────────────────┐ │
│  │  FastAPI / Uvicorn    │─────▶│  Groq Cloud API  │ │
│  │  Python :8000         │      │  HTTPS :443      │ │
│  │  (+ Embeddings,       │      └──────────────────┘ │
│  │   Reranker on CPU)    │                           │
│  └──────┬───────┬────────┘                           │
│         │       │                                    │
│  ┌──────▼──┐ ┌──▼──────────┐                        │
│  │Postgres │ │   Qdrant    │                        │
│  │Docker   │ │   Docker    │                        │
│  │:5432    │ │   :6333     │                        │
│  └─────────┘ └─────────────┘                        │
└─────────────────────────────────────────────────────┘
```

### 11.2 Service Configuration

| Service | Runtime | Port | Container | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Streamlit Frontend | Python (Tornado) | 8501 | No | `streamlit run app.py` |
| FastAPI Backend | Python 3.11 / Uvicorn | 8000 | No | `uvicorn main:app --port 8000` |
| PostgreSQL | PostgreSQL 16 | 5432 | **Yes** (Docker) | Bound to `127.0.0.1:5432` |
| Qdrant | Qdrant latest | 6333 | **Yes** (Docker) | Bound to `127.0.0.1:6333` |
| Groq Cloud API | External SaaS | 443 | N/A | HTTPS, requires `GROQ_API_KEY` |

### 11.3 Docker Compose Specification

```yaml
# docker-compose.yml (authoritative)
version: "3.9"
services:
  postgres:
    image: postgres:16
    ports:
      - "127.0.0.1:5432:5432"           # Loopback only (ADL-019)
    environment:
      POSTGRES_DB: medbridge
      POSTGRES_USER: medbridge_app
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}   # From .env, not hardcoded
    volumes:
      - pgdata:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "127.0.0.1:6333:6333"           # Loopback only (ADL-019)
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  pgdata:
  qdrant_data:
```

### 11.4 Environment Variables

| Variable | Required | Description |
| :--- | :---: | :--- |
| `GROQ_API_KEY` | Yes | API key for Groq Cloud LLM inference |
| `OPENAI_API_KEY` | Conditional | API key for GPT-4o-mini (if used as fallback) |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password (not hardcoded in compose) |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://medbridge_app:{password}@127.0.0.1:5432/medbridge` |
| `QDRANT_URL` | Yes | `http://127.0.0.1:6333` |
| `LLM_MODEL` | Yes | Model identifier (e.g., `llama-3.1-8b-instant` or `gpt-4o-mini`) |

### 11.5 Local Compute Requirements

| Resource | Requirement |
| :--- | :--- |
| **GPU** | Not required. All LLM inference is offloaded to Groq Cloud API |
| **CPU** | BGE embedding model (~33M params) and cross-encoder reranker run locally on CPU |
| **RAM** | ≥ 8 GB (PostgreSQL + Qdrant + Python process + embedding models) |
| **Disk** | ≥ 5 GB (Docker images + Qdrant vectors + PostgreSQL data) |

> [!NOTE]
> **Conflict Resolved (ADL-021):** The deployment diagram labels the frontend-to-backend connection as "HTTPS". For local development on localhost, this document standardizes on plain **HTTP** (`http://localhost:8000`). TLS termination via a reverse proxy is a production concern only.

---

## 12. Assumptions

| ID | Assumption | Impact If Wrong |
| :--- | :--- | :--- |
| **ASM-01** | No user authentication is required. Sessions are identified by UUID only. Any user with the session ID can access the conversation. | Security vulnerability in any non-academic deployment |
| **ASM-02** | The Groq Cloud API free tier provides sufficient rate limits for development and evaluation (~30 RPM for Llama 3.1 8B). | Pipeline will be throttled; may need to switch to paid tier or OpenAI |
| **ASM-03** | A single instance of Llama 3.1 8B (instruction-tuned) can reliably perform all 4 pipeline tasks (extraction, gate classification, evidence evaluation, response generation) via prompt engineering alone, without fine-tuning. | Gate accuracy will be insufficient; may need task-specific models or few-shot examples |
| **ASM-04** | Three clinical guideline sources (AHA/ACC 2025, ESC/ESH 2024, MedlinePlus) provide sufficient coverage for adult hypertension queries in the evaluation benchmark. | Out-of-domain queries will trigger excessive ABSTAIN rates |
| **ASM-05** | The fastembed library supports client-side BM25 sparse vector generation compatible with Qdrant's named sparse vector format. | Hybrid search degrades to dense-only retrieval |
| **ASM-06** | PostgreSQL and Qdrant Docker containers can run simultaneously on a typical developer laptop (8+ GB RAM) without resource contention. | Memory pressure causes OOM kills or container restarts |
| **ASM-07** | Patient inputs are in English and relate to adult hypertension management. The system does not handle pediatric, pregnancy-related, or non-cardiovascular queries. | Increased ABSTAIN/ESCALATE rates for off-topic queries |
| **ASM-08** | The `soft_ask_count` threshold of 2 is a reasonable limit before falling back to `GENERALIZE`. This number may need tuning based on user study results. | Patients may be frustrated if asked too many questions, or context may be insufficient if too few |
| **ASM-09** | The Deterministic State Projector can handle all event types with simple merge/overwrite rules without requiring a complex CRDT or conflict resolution system. | Edge cases in conflicting events (e.g., contradictory BP readings) may produce incorrect snapshots |
| **ASM-10** | The cross-encoder reranker can run within acceptable latency (< 1.5s) on CPU when limited to 20 candidate pairs using a lightweight model (ms-marco-MiniLM-L-6-v2). | End-to-end latency exceeds 8-second target |

---

## 13. Constraints

| ID | Constraint | Source |
| :--- | :--- | :--- |
| **CON-01** | **Academic semester timeline.** The system must be implementable, deployable, and evaluable within a single university semester. | Project scope |
| **CON-02** | **Zero local GPU.** No local GPU is available. All heavy LLM inference must use cloud APIs (Groq / OpenAI). Local CPU handles only embedding and reranking. | Deployment diagram |
| **CON-03** | **Single-node deployment.** No Kubernetes, no multi-node orchestration. All services run on one machine via Docker Compose + local Python processes. | Deployment diagram |
| **CON-04** | **External LLM dependency.** The system critically depends on Groq or OpenAI API availability. A cloud outage disables the entire AI pipeline. | Architecture |
| **CON-05** | **Clinical guideline currency.** The knowledge base is a static snapshot of AHA/ACC 2025 and ESC/ESH 2024 guidelines. It does not auto-update when new guidelines are published. | Knowledge sources |
| **CON-06** | **No fine-tuning.** The LLM is used via prompt engineering and structured output only. No model weights are modified. | Spec |
| **CON-07** | **Python-only backend.** The entire backend and AI pipeline must be implemented in Python 3.11+. No polyglot services. | Tech stack |
| **CON-08** | **RAG framework: Raw SDK.** The retrieval pipeline uses direct `qdrant-client` + `fastembed` + `sentence-transformers` SDK calls, not a higher-level framework like LlamaIndex. | ADL-006 resolution |

> [!NOTE]
> **Conflict Resolved (ADL-006):** The requirements document mentioned LlamaIndex, but all diagrams show custom retrieval components. This document standardizes on the **raw SDK approach** (`qdrant-client` + `fastembed` + `sentence-transformers`) as the diagrams represent the actual intended architecture.

---

## 14. Risks

### 14.1 Risk Register

| ID | Risk | Likelihood | Impact | Severity | Mitigation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RSK-01** | **Groq API rate limiting or outage** halts all LLM inference, rendering the system non-functional | Medium | Critical | 🔴 | Implement provider failover (Groq → OpenAI). Configure retry with exponential backoff. Cache deterministic gate responses for repeated identical queries. |
| **RSK-02** | **LLM structured output failures** — Llama 3.1 8B produces malformed JSON that cannot be parsed after retries | High | High | 🟠 | Resilient LLM Wrapper with 3-attempt retry, JSON repair, and deterministic fallback to GENERALIZE/ESCALATE (never HTTP 500). |
| **RSK-03** | **PHI/PII leakage to cloud providers** — patient health data transmitted to Groq/OpenAI without anonymization | High | Critical | 🔴 | Documented as an architectural gap (ADL-017). Mitigation: add PII scrubbing layer in a future phase. For the semester, use synthetic test data only and document the limitation. |
| **RSK-04** | **Prompt injection attacks** — adversarial patient inputs manipulate gate classifications, overriding safety routing | Medium | Critical | 🔴 | Implement XML-delimited prompt structure separating system instructions from untrusted user input. Add input sanitization heuristics. Document as a known attack surface. |
| **RSK-05** | **Gate classification accuracy** — Context Gate or Evidence Gate misclassifies, leading to unsafe ANSWER when ABSTAIN/ESCALATE was correct | Medium | Critical | 🔴 | Validate via MedBridge-AQ benchmark (200+ vignettes). Ablation studies comparing gate presence vs. absence. Conservative gate thresholds (prefer false-positive SOFT-ASK over false-negative PROCEED). |
| **RSK-06** | **Cross-encoder CPU latency** exceeds acceptable limits, causing end-to-end response time > 8 seconds | Medium | Medium | 🟡 | Use lightweight reranker (ms-marco-MiniLM-L-6-v2, 22M params). Offload to background thread. Benchmark during development to validate. |
| **RSK-07** | **Knowledge base coverage gaps** — patient asks about a drug or condition not covered by AHA/ACC/ESC/MedlinePlus | High | Medium | 🟡 | Evidence Gate routes to ABSTAIN or GENERALIZE. Add MedlinePlus as a broader fallback source. Document the domain boundary clearly in the UI. |
| **RSK-08** | **Event Sourcing snapshot drift** — if the Deterministic State Projector has bugs, the snapshot diverges from the event log, causing incorrect context for all downstream gates | Low | High | 🟠 | Unit test all projection rules exhaustively. Implement a snapshot reconstruction validator that replays the event log and compares against the stored snapshot. |
| **RSK-09** | **Evaluation benchmark bias** — MedBridge-AQ vignettes do not adequately represent real patient query distributions | Medium | High | 🟠 | Target Cohen's κ ≥ 0.6. Include adversarial examples (RABBITS). Compare against multiple baselines. Document limitations in the evaluation chapter. |
| **RSK-10** | **Single-point-of-failure deployment** — all services on one node; any service crash (PostgreSQL, Qdrant, Uvicorn) takes down the entire system | High | Medium | 🟡 | Acceptable for semester scope. Document as a production gap. Docker Compose restart policies (`restart: unless-stopped`) provide basic resilience. |

### 14.2 Unmitigated Architectural Gaps (Acknowledged)

The following items from the ADL are acknowledged as gaps that will **not** be fully resolved in the semester implementation. They are documented here for transparency and future work.

| ADL | Gap | Semester Status |
| :--- | :--- | :--- |
| ADL-017 | PII Anonymization Layer | Deferred. Use synthetic data for evaluation. Document as limitation. |
| ADL-018 | Prompt Injection Defense | Partially mitigated via XML isolation in prompt structure. Full adversarial testing deferred. |
| ADL-019 | Database Authentication & Network Binding | Mitigated by binding to `127.0.0.1` in Docker Compose. Full auth configuration documented but not enforced. |

---

## Appendix A — Conflict Resolution Register

This appendix maps every conflict identified in the [Architecture Decision Log](architecture_decision_log.md) to its resolution in this document.

| ADL | Conflict | Resolution in This Document |
| :--- | :--- | :--- |
| ADL-001 | No deterministic state projector | Added as mandatory component (§7.1, §9.1) |
| ADL-002 | No emergency fast-path | Added Emergency Triage Classifier (§6.4, §7.1) |
| ADL-003 | No ingestion service spec | Added IngestionWorker component (§7.1, §6.6) |
| ADL-004 | No LLM parse failure handler | Added Resilient LLM Wrapper (§7.1, §9.8) |
| ADL-005 | PostgreSQL Docker contradiction | Standardized: both PostgreSQL + Qdrant in Docker Compose (§11.3) |
| ADL-006 | LlamaIndex vs. raw SDK | Standardized: raw SDK (qdrant-client + fastembed) (§13, CON-08) |
| ADL-007 | Streamlit mislabelled as Node.js | Standardized: Streamlit (Python :8501) (§7.1, §11.2) |
| ADL-008 | Inconsistent action names | Standardized 5-action enum (§1.3) |
| ADL-009 | REST API schemas undefined | Defined in §7.2 and §8 |
| ADL-010 | LLM output schemas undefined | Defined in §9.2–9.6 |
| ADL-011 | Qdrant schema undefined | Defined in §8.5 |
| ADL-012 | PostgreSQL DDL undefined | Defined in §8.1–8.4 |
| ADL-013 | No loop-breaker for soft_ask ≥ 2 | Added as workflow §6.3 and gate logic §9.3 |
| ADL-014 | No DB read for session state | Added as Step 2.5 in workflow §6.1 |
| ADL-015 | ABSTAIN/ESCALATE through LLM | Restricted LLM Call 4 to ANSWER/GENERALIZE only (§6.5, §9.6) |
| ADL-016 | No inline citation attribution | Added inline markers to ResponseGeneratorOutput (§9.6) |
| ADL-017 | PHI/PII to cloud APIs | Acknowledged gap; deferred with synthetic data mitigation (§14.2) |
| ADL-018 | Prompt injection vulnerability | Partially mitigated via XML isolation (§9.6, §14.2) |
| ADL-019 | Unsecured database ports | Bound to 127.0.0.1 in Docker Compose (§11.3) |
| ADL-020 | CPU reranker blocks event loop | Specified lightweight model + threadpool offload (§9.4) |
| ADL-021 | HTTPS on localhost | Standardized: HTTP for local dev (§11.2) |
| ADL-022 | Sparse vector generation unspecified | Specified fastembed client-side generation (§8.5, §9.4) |
| ADL-023 | Context Gate dual responsibility | Moved query reformulation to LLM Call 1 (§9.2, §9.3) |
| ADL-024 | Evidence Gate criteria undefined | Defined decision matrix (§9.5) |
| ADL-025 | XML-Isolation template undefined | Defined canonical template (§9.6) |
| ADL-026 | Gate temperature unspecified | Defined inference parameters (§9.7) |

---

## Appendix B — Errata (2026-10)

> **Date:** 2026-10-10. This appendix is append-only: the sections above remain the consolidated design-time context of 2026-08-15. Each erratum states what no longer holds for the implemented system and points to the decision that changed it (`docs/architecture_decision_log.md`, ADL-027 – ADL-058), to the constitution amendments (`docs/project_constitution.md` §14) or to the implementation part of the specification (`docs/technical_specification.md`, Part VII). The architecture itself (Dual Answerability Gate, five actions, deterministic fast-path, loop-breaker at 2, template bypass, XML isolation, event sourcing with append-only logs, never-throw orchestrator) is unchanged.
>
> The only in-place edit to this document is in Appendix A: the link to the Architecture Decision Log pointed to a machine-specific absolute path and now points to the repository file `architecture_decision_log.md`.

| ID | Section | Statement in this document | Current state | Reference |
| :--- | :--- | :--- | :--- | :--- |
| MPC-E01 | §1.1 – §1.2, §5 | The system serves hypertension patients | Scope is cardiovascular care centred on hypertension (heart failure, atrial fibrillation, coronary disease, lipids, CKD, diabetes-related risk, hypertensive disorders of pregnancy, prevention); non-cardiovascular and non-medical topics are out of domain | Constitution A-8 |
| MPC-E02 | §2.1 | Chat interface only | Additional patient features: streamed stage progress (SSE), regenerate, feedback, "Why this answer?" panel, patient context card, BP log with ACC/AHA categories, clinician summary export, right to erasure | ADL-053 – ADL-055, ADL-032 |
| MPC-E03 | §2.3 FR-11, FR-14 | Three knowledge sources; PDF ingestion only | 27 documents listed in `data/corpus_manifest.json` (PDF and HTML) with licence and SHA-256 pins (`reports/corpus_stats.md`) | ADL-046, ADL-047 |
| MPC-E04 | §3.1 NFR-01, NFR-02 | ≤ 8 s (PROCEED path) and ≤ 3 s (SOFT-ASK), assuming cloud inference | With local 4-bit models these are design-time targets, not guarantees. The round-1 model-selection run measured a turn p50 of 10.0 s and p95 of 23.4 s for the selected model over all dev-split turns (`evaluation/results/model_selection_round1_prefix/model_selection.md`). The before/after performance table is recorded in `reports/performance/latency_table.md` (same model before and after; ADL-049) | ADL-049, ADL-057 |
| MPC-E05 | §3.1 NFR-03 | Emergency fast-path ≤ 500 ms | Still deterministic and LLM-free; unit tests enforce under 5 ms on typical messages (Non-Negotiable 1) | ADL-045 |
| MPC-E06 | §3.2 NFR-08 | Groq availability fallback | Local Ollama with retries and the unchanged spec fallbacks; Groq is optional and disabled in local-only mode | ADL-039, ADL-040 |
| MPC-E07 | §3.3 NFR-11 | PII handling is a documented gap; anonymization deferred | Implemented: deterministic redaction before any LLM call, log line or write; encryption at rest; crypto-shredding erasure; evaluation in `reports/privacy_redaction_metrics.md` | ADL-030 – ADL-035 |
| MPC-E08 | §3.4 NFR-12 – NFR-15 | Benchmark, κ, RABBITS and baselines planned | MedBridge-AQ rebuilt with 275 vignettes; RABBITS covers 189 brand → generic pairs (65 Indian) from the project's own lexicon rather than live RxNorm queries; baselines and ablations implemented as conditions C0 – C5 (plus V1/V2) on MedBridge-HalluBench v1; a 50-item human annotation sheet supports κ. Live benchmark and study results are recorded (README "Results", rendered from `reports/` and `evaluation/results/`; ADL-049, ADL-056, ADL-059); the human–judge κ is pending until the team labels the exported sheet | ADL-044, ADL-056, ADL-059 |
| MPC-E09 | §4.1 | External cloud LLM (Groq / OpenAI) | Local Ollama server on loopback; no outbound LLM connection in local-only mode | ADL-040 |
| MPC-E10 | §4.2 | Authentication, PII anonymization and multilingual support out of scope | Signed session tokens and admin login are in scope; PII redaction is enforced; the emergency fast-path also recognises Hinglish / Indian-English red-flag phrasing (the rest of the pipeline remains English-only) | ADL-030, ADL-033, ADL-034, ADL-045 |
| MPC-E11 | §5 (ASM-01 note) | Any user with the session URL can resume a session | Every session route requires a bearer token signed for that session; the token is never placed in the URL. An administrator role exists for the dashboard; "clinician mode" in the web UI is a presentation setting only | ADL-033, ADL-034, ADL-053 |
| MPC-E12 | §6 | Workflows persist state between extraction and Gate 1 and contain no privacy or small-talk step | Privacy redaction first, then the fast-path on the raw message; small-talk check; in-memory projection; optional speculative retrieval during Gate 1; citation validation; optional Gate 3; one persistence step per turn. After a SOFT-ASK, the gates judge the original question plus the follow-up details | ADL-027 – ADL-029, ADL-036, ADL-043; Spec Part VII §3 |
| MPC-E13 | §7 | Component registry | New components: privacy layer, web SPA and dashboard, deterministic NLP, complementary features, evaluation package, V1/V2 verification gates, benchmark harnesses | `docs/module_map.md` |
| MPC-E14 | §8 | Data entities | Added `audit_logs.trace`, `sessions.erased_at`, `sessions.legacy_plaintext_rows`, `session_keys`, `feedback`; patient-derived fields are encrypted; snapshot writers lock and re-project | ADL-031, ADL-032, ADL-036, ADL-038 |
| MPC-E15 | §9.2 – §9.6 | Prompt and schema descriptions | Rule-based few-shot prompts, JSON-schema structured output, Gate 1 `missing_fields` from a fixed vocabulary, generator cites `[n]` indices validated by the server | ADL-039, ADL-041, ADL-042, ADL-028 |
| MPC-E16 | §9.7 | Generator seed "—" | The generator also uses seed 42 (temperature 0.3, top_p 0.95) for reproducible evaluation; gates unchanged at 0.0 / 42 | ADL-039 |
| MPC-E17 | §9.8 | Permanent failure → GENERALIZE (calls 1 – 3) or ESCALATE (call 4) | Implemented fallbacks follow the specification: extractor → no events; Gate 1 → PROCEED; Gate 2 → GENERALIZE; generator → pre-vetted GENERALIZE template. Truncated output is never repaired | ADL-036, ADL-039 |
| MPC-E18 | §10, §11 | PostgreSQL on :5432; Groq environment variables; no GPU needed | PostgreSQL on host port 5433; new required secrets (`ENCRYPTION_MASTER_KEY`, `SESSION_TOKEN_SECRET`), optional `ADMIN_PASSWORD`, `EVAL_DATABASE_URL` for evaluation; local LLM inference benefits from a GPU (Apple Silicon or the RTX 4050 demonstration laptop) | Spec Part VII §10 – §11 |
| MPC-E19 | §12 | ASM-01, ASM-02, ASM-04, ASM-07 | ASM-01 superseded (tokens); ASM-02 obsolete (no Groq dependency); ASM-04 superseded (27-document corpus); ASM-07 superseded (wider cardiovascular scope; paediatric and pregnancy dosing questions are handled by explicit Gate 2 rules) | ADL-033, ADL-040, ADL-041, ADL-046 |
| MPC-E20 | §12 ASM-03 | One model per pipeline without few-shot examples | Still no fine-tuning; prompts now include few-shot examples and each role may use its own model | ADL-039, ADL-041 |
| MPC-E21 | §13 CON-02, CON-04, CON-05 | No local GPU; dependency on cloud LLM APIs; static AHA/ACC + ESC knowledge base | Local inference on the user's machine; no cloud dependency at runtime; the knowledge base is still a static, versioned snapshot, now of 27 documents | ADL-040, ADL-046 |
| MPC-E22 | §14.1 | Risk register | RSK-01 becomes "local LLM server unavailable" (mitigated by warm-up, readiness reporting and the spec fallbacks); RSK-03 mitigated (redaction, local-only inference); RSK-04 mitigated further (full isolation, tag neutralisation, adversarial test suite); RSK-05 is measured by the HalluBench study (pending); RSK-07 reduced by corpus expansion (heart failure remains the thinnest topic) | ADL-030, ADL-040, ADL-041, ADL-046, ADL-049, ADL-056 |
| MPC-E23 | §14.2 | ADL-017 deferred; ADL-018 partially mitigated | ADL-017 implemented (ADL-030, ADL-040); ADL-018 strengthened and tested (ADL-041, `tests/adversarial/`); ADL-019 unchanged (loopback binding, credentials from the environment) | ADL-030, ADL-040, ADL-041 |
| MPC-E24 | Appendix A | Register ends at ADL-026 | Decisions ADL-027 – ADL-058 are listed in the summary table of `docs/architecture_decision_log.md` | ADL-027 – ADL-058 |
