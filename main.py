from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional, List
import hashlib

# ========= 数据库配置：用一个本地文件 database.db =========
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, echo=False)

app = FastAPI(title="Validation Platform MVP")

# 允许前端跨域访问（以后你做网页 / 小程序用得到）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 数据模型 ======================

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    name: str
    password_hash: str
    role: str = "user"  # "user" 或 "creator"


class UserCreate(SQLModel):
    email: str
    name: str
    password: str
    role: str = "user"  # 传 "creator" 就是创业者账号


class UserLogin(SQLModel):
    email: str
    password: str


class UserOut(SQLModel):
    id: int
    email: str
    name: str
    role: str


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    creator_id: int = Field(index=True)
    title: str
    description: str
    target_audience: str
    questions: str  # 用换行符把问题拼成一串
    reward_note: Optional[str] = None
    budget: int  # 预算（单位：元）
    status: str = "active"  # "active" 或 "closed"


class ProjectCreate(SQLModel):
    creator_id: int
    title: str
    description: str
    target_audience: str
    questions: List[str]
    reward_note: Optional[str] = None
    budget: int


class ProjectOut(SQLModel):
    id: int
    creator_id: int
    title: str
    description: str
    target_audience: str
    questions: List[str]
    reward_note: Optional[str]
    budget: int
    status: str


class Response(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    user_id: int = Field(index=True)
    interest_level: int  # 1-5
    answers: str         # 同样用换行符拼起来
    price_min: Optional[int] = None
    price_max: Optional[int] = None


class ResponseCreate(SQLModel):
    user_id: int
    interest_level: int
    answers: List[str]
    price_min: Optional[int] = None
    price_max: Optional[int] = None


class ProjectStats(SQLModel):
    project_id: int
    total_responses: int
    avg_interest: Optional[float]
    avg_price_min: Optional[float]
    avg_price_max: Optional[float]


# ====================== 工具函数 ======================

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


def hash_password(password: str) -> str:
    """非常简单的密码哈希，够 MVP 用，上线前再换更安全的方案"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ====================== 启动时建表 ======================

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# ====================== 用户注册 / 登录 ======================

@app.post("/register", response_model=UserOut)
def register(user: UserCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == user.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user = User(
        email=user.email,
        name=user.name,
        password_hash=hash_password(user.password),
        role=user.role,
    )
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return UserOut.model_validate(db_user)


@app.post("/login")
def login(data: UserLogin, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == data.email)).first()
    if not user or user.password_hash != hash_password(data.password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    # 不搞复杂鉴权，先返回 user_id，前端后面自己带着用
    return {"user_id": user.id, "role": user.role, "name": user.name}


# ====================== 创业者创建验证项目 ======================

@app.post("/projects", response_model=ProjectOut)
def create_project(p: ProjectCreate, session: Session = Depends(get_session)):
    creator = session.get(User, p.creator_id)
    if not creator or creator.role != "creator":
        raise HTTPException(status_code=400, detail="creator_id is invalid or not a creator")

    questions_text = "\n".join(p.questions)
    proj = Project(
        creator_id=p.creator_id,
        title=p.title,
        description=p.description,
        target_audience=p.target_audience,
        questions=questions_text,
        reward_note=p.reward_note,
        budget=p.budget,
        status="active",
    )
    session.add(proj)
    session.commit()
    session.refresh(proj)
    return ProjectOut(
        id=proj.id,
        creator_id=proj.creator_id,
        title=proj.title,
        description=proj.description,
        target_audience=proj.target_audience,
        questions=proj.questions.split("\n"),
        reward_note=proj.reward_note,
        budget=proj.budget,
        status=proj.status,
    )


# ====================== 用户侧：获取项目列表 ======================

@app.get("/projects/active", response_model=list[ProjectOut])
def list_active_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).where(Project.status == "active")).all()
    result = []
    for proj in projects:
        result.append(
            ProjectOut(
                id=proj.id,
                creator_id=proj.creator_id,
                title=proj.title,
                description=proj.description,
                target_audience=proj.target_audience,
                questions=proj.questions.split("\n"),
                reward_note=proj.reward_note,
                budget=proj.budget,
                status=proj.status,
            )
        )
    return result


# ====================== 用户提交反馈 ======================

@app.post("/projects/{project_id}/respond")
def respond_to_project(
    project_id: int,
    r: ResponseCreate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project or project.status != "active":
        raise HTTPException(status_code=404, detail="Project not found or inactive")

    user = session.get(User, r.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    # 防止重复答题
    existing = session.exec(
        select(Response).where(
            Response.project_id == project_id,
            Response.user_id == r.user_id
        )
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already responded to this project")

    resp = Response(
        project_id=project_id,
        user_id=r.user_id,
        interest_level=r.interest_level,
        answers="\n".join(r.answers),
        price_min=r.price_min,
        price_max=r.price_max,
    )
    session.add(resp)
    session.commit()
    return {"ok": True}


# ====================== 创业者查看项目统计 ======================

@app.get("/projects/{project_id}/stats", response_model=ProjectStats)
def project_stats(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    responses = session.exec(
        select(Response).where(Response.project_id == project_id)
    ).all()
    total = len(responses)

    if total == 0:
        return ProjectStats(
            project_id=project_id,
            total_responses=0,
            avg_interest=None,
            avg_price_min=None,
            avg_price_max=None,
        )

    avg_interest = sum(r.interest_level for r in responses) / total

    price_min_values = [r.price_min for r in responses if r.price_min is not None]
    price_max_values = [r.price_max for r in responses if r.price_max is not None]

    avg_price_min = (
        sum(price_min_values) / len(price_min_values) if price_min_values else None
    )
    avg_price_max = (
        sum(price_max_values) / len(price_max_values) if price_max_values else None
    )

    return ProjectStats(
        project_id=project_id,
        total_responses=total,
        avg_interest=avg_interest,
        avg_price_min=avg_price_min,
        avg_price_max=avg_price_max,
    )

# ====================== 网站运行状态 ======================

@app.get("/health")
def health():
    return {"status": "ok"}