#!/usr/bin/env python3
"""
实体抽取器
从帖子内容中抽取人/组织/品牌/产品/事件/地点实体
基于规则 + jieba + 可选 LLM 增强
"""
import re
import random
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

# 尝试导入 jieba，如果没有则使用简单规则回退
try:
    import jieba
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


# ── 内置词库 ──────────────────────────────

# 品牌词库（可扩展）
BRAND_KEYWORDS = {
    "华为", "苹果", "小米", "三星", "OPPO", "vivo", "一加", "荣耀",
    "特斯拉", "比亚迪", "蔚来", "小鹏", "理想", "奔驰", "宝马", "奥迪",
    "茅台", "五粮液", "星巴克", "喜茶", "奈雪", "蜜雪冰城",
    "淘宝", "京东", "拼多多", "抖音", "快手", "小红书", "微博", "知乎",
    "腾讯", "阿里", "阿里巴巴", "字节跳动", "百度", "美团", "滴滴",
    " Nike", "Adidas", "李宁", "安踏",
}

# 组织机构词库
ORG_KEYWORDS = {
    "武汉大学", "北京大学", "清华大学", "复旦大学", "浙江大学",
    "中科院", "中国工程院", "卫健委", "教育部", "工信部",
    "国务院", "发改委", "市场监管局", "消协",
    "联合国", "世界卫生组织", "世界银行",
}

# 地点词库
LOCATION_KEYWORDS = {
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
    "西安", "重庆", "天津", "苏州", "长沙", "郑州", "青岛",
    "美国", "日本", "韩国", "英国", "德国", "法国", "俄罗斯",
}

# 产品模式
PRODUCT_PATTERNS = [
    r'iPhone\s*\d+', r'iPad\s*\w*', r'MacBook\s*\w*',
    r'华为\s*[P|M]\d+', r'小米\s*\d+', r'三星\s*Galaxy\s*\w+',
]

# 事件模式（时间 + 关键词）
EVENT_PATTERNS = [
    r'\d{1,2}·\d{1,2}', r'315', r'双十一', r'双11', r'618',
    r'春节', r'国庆', r'五一', r'元旦',
]


@dataclass
class ExtractedEntity:
    """抽取出的实体"""
    name: str
    entity_type: str
    confidence: float = 1.0
    source_post_id: str = ""
    position: Tuple[int, int] = (0, 0)  # 在文本中的位置


class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self, custom_brands: Optional[Set[str]] = None,
                 custom_orgs: Optional[Set[str]] = None):
        self.brands = BRAND_KEYWORDS | (custom_brands or set())
        self.orgs = ORG_KEYWORDS | (custom_orgs or set())
        self.locations = LOCATION_KEYWORDS
        
        # 加载 jieba 自定义词库（如果有）
        if HAS_JIEBA:
            for word in self.brands | self.orgs | self.locations:
                jieba.add_word(word, freq=1000)
    
    def extract_from_text(self, text: str, post_id: str = "") -> List[ExtractedEntity]:
        """从单条文本中抽取实体"""
        entities = []
        
        # 1. 规则匹配（品牌/组织/地点）
        entities.extend(self._extract_by_keywords(text, post_id))
        
        # 2. 正则匹配（产品/事件）
        entities.extend(self._extract_by_patterns(text, post_id))
        
        # 3. jieba 词性标注（人名/机构名/地名）
        if HAS_JIEBA:
            entities.extend(self._extract_by_jieba(text, post_id))
        
        # 去重（按名称+类型）
        seen = set()
        unique = []
        for e in entities:
            key = (e.name, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        return unique
    
    def extract_from_posts(self, posts: List[Dict]) -> List[ExtractedEntity]:
        """从多条帖子中批量抽取实体"""
        all_entities = []
        for post in posts:
            text = post.get("content", "")
            post_id = post.get("post_id", "")
            entities = self.extract_from_text(text, post_id)
            all_entities.extend(entities)
        return all_entities
    
    def _extract_by_keywords(self, text: str, post_id: str) -> List[ExtractedEntity]:
        """基于关键词库匹配"""
        entities = []
        
        for brand in self.brands:
            if brand in text:
                idx = text.find(brand)
                entities.append(ExtractedEntity(
                    name=brand, entity_type="brand",
                    confidence=0.9, source_post_id=post_id,
                    position=(idx, idx + len(brand))
                ))
        
        for org in self.orgs:
            if org in text:
                idx = text.find(org)
                entities.append(ExtractedEntity(
                    name=org, entity_type="organization",
                    confidence=0.9, source_post_id=post_id,
                    position=(idx, idx + len(org))
                ))
        
        for loc in self.locations:
            if loc in text:
                idx = text.find(loc)
                entities.append(ExtractedEntity(
                    name=loc, entity_type="location",
                    confidence=0.85, source_post_id=post_id,
                    position=(idx, idx + len(loc))
                ))
        
        return entities
    
    def _extract_by_patterns(self, text: str, post_id: str) -> List[ExtractedEntity]:
        """基于正则模式匹配"""
        entities = []
        
        # 产品匹配
        for pattern in PRODUCT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    name=match.group(), entity_type="product",
                    confidence=0.85, source_post_id=post_id,
                    position=(match.start(), match.end())
                ))
        
        # 事件匹配
        for pattern in EVENT_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    name=match.group(), entity_type="event",
                    confidence=0.8, source_post_id=post_id,
                    position=(match.start(), match.end())
                ))
        
        return entities
    
    def _extract_by_jieba(self, text: str, post_id: str) -> List[ExtractedEntity]:
        """基于 jieba 词性标注"""
        entities = []
        
        words = pseg.cut(text)
        for word, flag in words:
            if len(word) < 2:
                continue
            
            # 人名
            if flag == 'nr':  # 人名
                entities.append(ExtractedEntity(
                    name=word, entity_type="person",
                    confidence=0.7, source_post_id=post_id,
                ))
            # 机构名
            elif flag == 'nt':  # 机构名
                entities.append(ExtractedEntity(
                    name=word, entity_type="organization",
                    confidence=0.75, source_post_id=post_id,
                ))
            # 地名
            elif flag == 'ns':  # 地名
                entities.append(ExtractedEntity(
                    name=word, entity_type="location",
                    confidence=0.75, source_post_id=post_id,
                ))
        
        return entities
    
    def extract_persons_from_mentions(self, posts: List[Dict]) -> List[ExtractedEntity]:
        """从 @提及 中抽取人名"""
        entities = []
        mention_pattern = re.compile(r'@([\w\u4e00-\u9fff]+)')
        
        for post in posts:
            text = post.get("content", "")
            post_id = post.get("post_id", "")
            for match in mention_pattern.finditer(text):
                name = match.group(1)
                entities.append(ExtractedEntity(
                    name=name, entity_type="person",
                    confidence=0.8, source_post_id=post_id,
                    position=(match.start(), match.end())
                ))
        
        return entities
