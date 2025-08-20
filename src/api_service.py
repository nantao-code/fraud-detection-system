"""
API服务模块
提供RESTful API接口用于模型调用
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

from model_deployer import ModelDeployer
from unified_config import UnifiedConfig


# 请求和响应模型
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="特征字典")
    model_name: str = Field(default="fraud_detection", description="模型名称")
    model_version: Optional[str] = Field(default="latest", description="模型版本")

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="数据列表")
    model_name: str = Field(default="fraud_detection", description="模型名称")
    model_version: Optional[str] = Field(default="latest", description="模型版本")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="预测结果")
    probability: float = Field(..., description="预测概率")
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本")
    timestamp: str = Field(..., description="预测时间")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="预测结果列表")
    summary: Dict[str, Any] = Field(..., description="统计摘要")

class ModelInfo(BaseModel):
    name: str = Field(..., description="模型名称")
    version: str = Field(..., description="模型版本")
    metrics: Dict[str, float] = Field(..., description="模型指标")
    created_at: str = Field(..., description="创建时间")
    feature_count: int = Field(..., description="特征数量")


class APIService:
    """API服务类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = UnifiedConfig(config_path)
        self.config = self.config_manager.to_dict()
        self.deployer = ModelDeployer(self.config)
        self.app = FastAPI(
            title="易受诈人群识别API",
            description="基于机器学习的易受诈人群识别服务",
            version="1.0.0"
        )
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """根路径"""
            return {
                "message": "易受诈人群识别API服务",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """获取模型列表"""
            try:
                models = self.deployer.list_models()
                return [
                    ModelInfo(
                        name=m['name'],
                        version=m['version'],
                        metrics=m['metrics'],
                        created_at=m['created_at'],
                        feature_count=len(m.get('feature_names', []))
                    )
                    for m in models
                ]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """单条预测"""
            try:
                # 加载模型
                model_info = self.deployer.load_model(
                    request.model_name, 
                    request.model_version
                )
                
                # 准备数据
                input_df = pd.DataFrame([request.features])
                
                # 预测
                results = self.deployer.predict(model_info, input_df)
                
                return PredictionResponse(
                    prediction=results['predictions'][0],
                    probability=results['probabilities'][0],
                    model_name=results['model_name'],
                    model_version=results['model_version'],
                    timestamp=results['timestamp']
                )
                
            except Exception as e:
                self.logger.error(f"预测失败: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def batch_predict(request: BatchPredictionRequest):
            """批量预测"""
            try:
                # 加载模型
                model_info = self.deployer.load_model(
                    request.model_name, 
                    request.model_version
                )
                
                # 准备数据
                input_df = pd.DataFrame(request.data)
                
                # 预测
                results = self.deployer.predict(model_info, input_df)
                
                # 构建响应
                predictions = [
                    PredictionResponse(
                        prediction=results['predictions'][i],
                        probability=results['probabilities'][i],
                        model_name=results['model_name'],
                        model_version=results['model_version'],
                        timestamp=results['timestamp']
                    )
                    for i in range(len(results['predictions']))
                ]
                
                # 统计摘要
                summary = {
                    'total_count': len(predictions),
                    'positive_count': sum(p.prediction for p in predictions),
                    'negative_count': len(predictions) - sum(p.prediction for p in predictions),
                    'avg_probability': np.mean([p.probability for p in predictions])
                }
                
                return BatchPredictionResponse(
                    predictions=predictions,
                    summary=summary
                )
                
            except Exception as e:
                self.logger.error(f"批量预测失败: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/models/{model_name}/info", response_model=ModelInfo)
        async def get_model_info(model_name: str, version: str = "latest"):
            """获取模型详细信息"""
            try:
                model_info = self.deployer.load_model(model_name, version)
                return ModelInfo(
                    name=model_info['model_name'],
                    version=model_info['model_version'],
                    metrics=model_info['metrics'],
                    created_at=model_info['created_at'],
                    feature_count=len(model_info['feature_names'])
                )
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
    
    def run(self):
        """运行API服务"""
        host = self.config.get('api.host', '0.0.0.0')
        port = self.config.get('api.port', 8000)
        reload = self.config.get('api.reload', False)
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload
        )


# 全局API实例
api_service = None

def create_api_service(config_path: str = "config.yaml") -> APIService:
    """创建API服务实例"""
    global api_service
    if api_service is None:
        api_service = APIService(config_path)
    return api_service


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动API服务")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    service = create_api_service(args.config)
    service.run()