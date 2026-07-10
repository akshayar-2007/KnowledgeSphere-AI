# Knowledge Management Portal

Centralized portal for institutional knowledge access across multiple stakeholders:
- Students
- Faculty
- Administrators
- Researchers

## Implemented Features
- Role-based authentication with secure password hashing.
- Stakeholder-specific dashboard (`student`, `faculty`, `administrator`, `researcher`).
- Domain management for organizing knowledge areas.
- PDF upload with:
  - Domain tagging
  - Audience visibility (`all`, `students`, `faculty`, `researchers`, `administrators`)
  - Optional keywords
- Semantic search with:
  - Domain filter
  - Audience filter
  - Ranking by similarity + keyword overlap
- Visibility-aware document library (users only see documents allowed for their role).
- Access control for preview/download endpoints.
- Activity logging for key actions (register/login/upload/domain changes).
- Offline-safe embedding fallback (app still runs if transformer model is unavailable online).
