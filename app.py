import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, Form, Query, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import io
import pandas as pd
from jobspy import scrape_jobs
from bs4 import BeautifulSoup
import requests
import time

#to use google gemini api
from google import genai

#to extract the env keys
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Career Pilot Ai",
    description="Beautiful, user-friendly UI and API on top of JobSpy job search",
    version="2.0.0",
)


gemClient = genai.Client(
    api_key=os.getenv('gemAPI')
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# Optional: import when available; keep a clear error if not installed


# ---------- Config ----------
SUPPORTED_SITES = ["indeed", "linkedin", "zip_recruiter", "glassdoor", "google", "bayt", "naukri"]
JOB_TYPES = ["fulltime", "parttime", "contract", "internship", "temporary", "volunteer", "apprenticeship"]
DEFAULT_RESULTS = 50

# API key auth (optional)
def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

ENABLE_API_KEY_AUTH = get_env_bool("ENABLE_API_KEY_AUTH", default=False)
API_KEY = os.getenv("API_KEY", "")

def require_api_key(x_api_key: Optional[str] = Query(None, alias="api_key")):
    if not ENABLE_API_KEY_AUTH:
        return True
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ---------- Mount static & templates ----------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Helpers ----------
def _safe_int(v: Optional[str], default: Optional[int]) -> Optional[int]:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except:
        return default

def _safe_float(v: Optional[str], default: Optional[float]) -> Optional[float]:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except:
        return default

def _scrape_with_params(params: Dict[str, Any]) -> pd.DataFrame:
    if scrape_jobs is None:
        raise RuntimeError("jobspy is not installed. Please `pip install jobspy`.")
    # Filter only known keys for jobspy; ignore Nones.
    kwargs = {
        "site_name": params.get("site_name"),
        "search_term": params.get("search_term"),
        "location": params.get("location") or None,
        "distance": params.get("distance"),
        "results_wanted": params.get("results_wanted") or DEFAULT_RESULTS,
        "hours_old": params.get("hours_old"),
        "job_type": params.get("job_type"),
    }
    # Remove None so jobspy doesn't get unexpected Nones
    kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}
    df = scrape_jobs(**kwargs)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df or [])
    # Normalize expected fields
    for col in ["title","company","location","via","date_posted","job_type","salary","min_salary","max_salary","currency","description","job_url","url","apply_link","apply_url"]:
        if col not in df.columns:
            df[col] = None
    # Derive a single apply_url
    def _pick_url(row):
        return row.get("apply_url") or row.get("apply_link") or row.get("job_url") or row.get("url")
    if "apply_url" not in df.columns:
        df["apply_url"] = None
    df["apply_url"] = df.apply(lambda r: _pick_url(r), axis=1)
    # Coerce date to string for template
    if "date_posted" in df.columns:
        try:
            df["date_posted"] = pd.to_datetime(df["date_posted"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return df


def _parse_common_params(
    site_name: Optional[List[str]] = None,
    job_type: Optional[List[str]] = None,
    search_term: Optional[str] = None,
    location: Optional[str] = None,
    results_wanted: Optional[int] = DEFAULT_RESULTS,
    hours_old: Optional[int] = None,
    distance: Optional[int] = None,
    sort_by: Optional[str] = "date_posted",
):
    # Validate and sanitize
    sites = [s for s in (site_name or SUPPORTED_SITES) if s in SUPPORTED_SITES]
    types = [t for t in (job_type or []) if t in JOB_TYPES]
    sort_by = sort_by if sort_by in ["date_posted", "title", "company", "salary"] else "date_posted"
    return {
        "site_name": sites or SUPPORTED_SITES,
        "job_type": types or None,
        "search_term": search_term or "",
        "location": (location or "").strip(),
        "results_wanted": results_wanted or DEFAULT_RESULTS,
        "hours_old": hours_old,
        "distance": distance,
        "sort_by": sort_by,
    }

#upcoming updates

#summerize the job description

# def llmSummerize(htmlDesc: str) -> str:
#     prompt = f"""
#     Generate a job summery description using: {htmlDesc}
#
#     IMPORTANT: Respond with ONLY the job description. No additional text, explanations, or markdown formatting.
#     """
#     for i in range(3):
#         try:
#             gem_response = gemClient.models.generate_content(
#                 model="gemini-2.5-flash", contents=prompt
#             )
#             gem_title = gem_response.text
#             break
#         except Exception as e:
#             gem_title = f"Gemini API failed: {str(e)}"
#
#     return gem_title


# scrap linkedin...
def extract_description(unique_jobs):
    description = []
    for job in unique_jobs:
        if job["site"] == "linkedin":
            url = job["job_url"]
            response = requests.get(url, headers=headers)

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the job details section by ID
            job_details = soup.find('div', class_='description__text description__text--rich')

            if job_details:
                #upcoming updates
                # store = llmSummerize(job_details.prettify())
                # description.append(store)
                description.append(job_details.prettify()) #after upcoming comment this line
            else:
                description.append("Job details section not found. The page structure might have changed.")
            time.sleep(2)
        else:
            description.append(f"Job details section Can't be fetched from {job["site"]}. The page can't be accessable.")

    return  description


# ---------- UI ROUTES ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Default form values
    ctx = dict(
        request=request,
        sites=SUPPORTED_SITES,
        job_types=JOB_TYPES,
        values=_parse_common_params(),
        jobs=[],
        count=0,
        error=None,
    )
    return templates.TemplateResponse("index.html", ctx)


@app.post("/search")
async def search_jobs_endpoint(
    request: Request,
    site_name: Optional[List[str]] = Form(None),
    job_type: Optional[List[str]] = Form(None),
    search_term: str = Form(""),
    location: str = Form(""),
    distance: int = Form(50),
    results_wanted: int = Form(20),
    hours_old: Optional[int] = Form(None),
    sort_by: Optional[str] = Form(None),
):
    try:
        # Handle job_type multi-select: run once per type, then merge
        all_jobs = []
        job_types_to_search = job_type if job_type else [None]

        for jt in job_types_to_search:
            jobs_df = scrape_jobs(
                site_name=site_name,
                search_term=search_term,
                location=location,
                distance=distance,
                results_wanted=results_wanted,
                hours_old=hours_old,
                job_type=jt,   # <-- one at a time
            )
            all_jobs.extend(jobs_df.to_dict("records"))

        # Deduplicate by job_url (if exists)
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            url = job.get("job_url")
            if url and url in seen:
                continue
            seen.add(url)
            unique_jobs.append(job)

        # Sorting
        if sort_by:
            if sort_by == "date":
                unique_jobs.sort(key=lambda x: x.get("date_posted") or "", reverse=True)
            elif sort_by == "title":
                unique_jobs.sort(key=lambda x: x.get("title") or "")
            elif sort_by == "company":
                unique_jobs.sort(key=lambda x: x.get("company") or "")
            elif sort_by == "salary":
                unique_jobs.sort(key=lambda x: x.get("salary") or "", reverse=True)

        if len(unique_jobs):
            description = extract_description(unique_jobs)
            for i in description:
                print(i, end="\n")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "jobs": unique_jobs,
            "count": len(unique_jobs),
            "search_term": search_term,
            "location": location,
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "jobs": [],
            "count": 0,
            "error": str(e),
        })



# ---------- API ROUTES ----------
class JobsQuery(BaseModel):
    site_name: Optional[List[str]] = Field(default=None, description="List of job sites to search")
    job_type: Optional[List[str]] = Field(default=None, description="List of job types")
    search_term: Optional[str] = Field(default="", description="Search keywords")
    location: Optional[str] = Field(default=None, description="Location to search")
    results_wanted: Optional[int] = Field(default=DEFAULT_RESULTS, ge=1, le=1000)
    hours_old: Optional[int] = Field(default=None, ge=1, description="Only jobs newer than X hours")
    distance: Optional[int] = Field(default=None, ge=1, description="Radius search in km or miles, depends on site")

@app.get("/api/jobs")
async def api_get_jobs(
    ok: bool = Depends(require_api_key),
    site_name: Optional[List[str]] = Query(default=None),
    job_type: Optional[List[str]] = Query(default=None),
    search_term: Optional[str] = Query(default=""),
    location: Optional[str] = Query(default=None),
    results_wanted: Optional[int] = Query(default=DEFAULT_RESULTS, ge=1, le=1000),
    hours_old: Optional[int] = Query(default=None, ge=1),
    distance: Optional[int] = Query(default=None, ge=1),
    format: Optional[str] = Query(default="json", regex="^(json|csv)$")
):
    params = _parse_common_params(site_name, job_type, search_term, location, results_wanted, hours_old, distance, "date_posted")
    try:
        df = _scrape_with_params(params)
        if format == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(output, media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=jobs.csv"})
        # default json
        return {"count": len(df), "jobs": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping jobs: {str(e)}")


@app.get("/export")
async def export(
    ok: bool = Depends(require_api_key) if ENABLE_API_KEY_AUTH else Depends(lambda: True),
    site_name: Optional[List[str]] = Query(default=None),
    job_type: Optional[List[str]] = Query(default=None),
    search_term: Optional[str] = Query(default=""),
    location: Optional[str] = Query(default=None),
    results_wanted: Optional[int] = Query(default=DEFAULT_RESULTS),
    hours_old: Optional[int] = Query(default=None),
    distance: Optional[int] = Query(default=None),
    format: Optional[str] = Query(default="csv", regex="^(csv|json)$"),
):
    params = _parse_common_params(site_name, job_type, search_term, location, results_wanted, hours_old, distance, "date_posted")
    try:
        df = _scrape_with_params(params)
        if format == "json":
            data = df.to_dict(orient="records")
            return JSONResponse(content=data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=jobs.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "ui": True, "api": True, "auth": ENABLE_API_KEY_AUTH}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)