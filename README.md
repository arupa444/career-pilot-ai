# CareerPilot AI 🚀

An intelligent, automated career assistant that finds the best jobs for you, tailors your resume, generates a personalized cover letter, applies for the job, and prepares you for interviews — all in one seamless pipeline.

---

## **Workflow / Process**

1. **AI Job Matcher**
   - Scans job boards & platforms (LinkedIn, Indeed, Google Jobs API alternatives) for relevant openings based on:
     - Skills
     - Experience
     - Career goals
     - Location preferences
     - Salary range

2. **Resume Tailor AI**
   - Parses the job description.
   - Modifies your resume (ATS-friendly) to highlight keywords and relevant skills.

3. **Cover Letter Generator**
   - Generates a personalized, role-specific cover letter using the job description & your resume.

4. **Application Bot**
   - Automates the job application process:
     - Auto-fills forms on job boards.
     - Attaches the tailored resume & cover letter.
     - Tracks application status.

5. **Interview Q&A Prep AI**
   - Analyzes the job role and company.
   - Generates tailored interview questions (technical + behavioral).
   - Provides suggested answers and learning resources.

---

## **Tech Stack**
- **Backend:** Python (FastAPI / Flask), Node.js (optional)
- **Frontend:** React / Next.js
- **Database:** PostgreSQL / MongoDB
- **AI Models:**
  - OpenAI GPT / LLaMA / Claude for text generation.
  - LangChain for orchestration.
  - spaCy / Transformers for NLP parsing.
- **Automation:**
  - Selenium / Playwright for web form filling.
  - Puppeteer (for Node.js automation).
- **Cloud & Deployment:**
  - AWS / Azure / GCP for hosting.
  - Docker for containerization.

---

## **Possible Data Sources**
Since **LinkedIn API** & **Google Jobs API** have restrictions:
- **LinkedIn API:** Official API is part of LinkedIn Marketing Developer Program (requires approval, not free).
- **Google Jobs API:** The Google Cloud Talent Solution API was discontinued for public use in 2021.
- **Alternatives:**
  - **Indeed API** (partner access required).
  - **Adzuna API** (free tier available).
  - **Jooble API** (free).
  - **Workable API**.
  - **Scraping** job listings with BeautifulSoup / Scrapy / Playwright.
  - Use **SerpAPI** or **Apify** for job search scraping as a service.

---

## **Modules Breakdown**
- **Module 1:** Job Finder Service
- **Module 2:** Resume Tailoring Engine
- **Module 3:** Cover Letter Generator
- **Module 4:** Auto Application Bot
- **Module 5:** Interview Preparation AI

---

## **Project Roadmap**
### **Phase 0: Research & Setup**
- Choose APIs or scraping methods for job search.
- Gather test resumes & job descriptions.
- Pick AI models for text generation & NLP.

### **Phase 1: Job Matching Engine**
- Implement job fetching from chosen sources.
- Add filtering logic (skills, salary, location).

### **Phase 2: Resume Tailor**
- Extract keywords from job descriptions.
- Modify resume templates dynamically.

### **Phase 3: Cover Letter Generation**
- Generate cover letters with LLM prompts.

### **Phase 4: Application Bot**
- Automate application submission process.

### **Phase 5: Interview Q&A AI**
- Generate role-specific interview questions.

---

## **License**
MIT License
