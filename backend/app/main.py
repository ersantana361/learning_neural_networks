from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import code_router, execute_router, health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown


app = FastAPI(
    title="Neural Networks Code Executor",
    description="Secure code execution backend for the Neural Networks learning platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://neural-networks.tunnel.ersantana.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(code_router)
app.include_router(execute_router)


@app.get("/")
async def root():
    return {"message": "Neural Networks Code Executor API", "docs": "/docs"}
