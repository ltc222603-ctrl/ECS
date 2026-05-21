from pathlib import Path
import torch
import sys
from pathlib import Path

# 当前文件：rasa/actions/xxx.py
BASE_DIR = Path(__file__).resolve().parent.parent  # 指向 rasa/

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.distill_bert_model import DistillBertStudent
from transformers import AutoTokenizer
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from sqlalchemy.orm import joinedload
from datetime import datetime
from typing import Any, Text, Dict, List
from .db import session_local
from .db_table_class import LogisticsCompany, OrderInfo, Logistics, LogisticsComplaint, LogisticsComplaintsRecord


# MODEL_PATH = BASE_DIR / "models" / "pruned_distill_bert_model.pt"
MODEL_PATH = BASE_DIR / "models" / "distill_bert_model_20260510_1811"
TOKENIZER_DIR = BASE_DIR / "models" / "distilbert-base-zh-cased"

device = torch.device("cpu")

# 全局加载（只执行一次）
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
model = DistillBertStudent().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# 按钮 → 等级映射
BUTTON_LEVEL_MAP = {
    # high（货物异常）
    "商品破损/丢失": "high",
    "货不对板": "high",

    # medium（配送服务异常）
    "拒绝配送": "medium",
    "擅自更改配送方式": "medium",
    "擅自退回包裹": "medium",
    "快递员态度恶劣": "medium",
    "未经授权代签": "medium",

    # low（物流信息异常）
    "虚假物流信息": "low",
    "长时间未更新物流": "low",
    "虚假签收": "low",
}

class GetLogisticsCompany(Action):
    """
    查询数据库，获取支持的快递公司列表
    """

    def name(self) -> Text:
        return "query_get_logistics_company"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        with session_local() as session:
            logistics_company = (session.query(LogisticsCompany).all())

        names = [company.company_name for company in logistics_company]
        result = "支持的快递有：\n" + "\n".join(f"- {name}" for name in names)

        dispatcher.utter_message(text=result)

        return []


class GetLogisticsInfo(Action):
    """
    查询指定订单id的物流信息
    """

    def name(self) -> Text:
        return "action_get_logistics_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #1、获取订单id
        order_id = tracker.slots.get("order_id")
        #2、查询物流信息，关联订单表、物流表、订单明细表
        with session_local() as session:
            order_info = (
                session.query(OrderInfo)
                .options(joinedload(OrderInfo.logistics))
                .options(joinedload(OrderInfo.order_detail))
                .filter(OrderInfo.order_id == order_id)
                .first()
            )
        #3、封装返回消息
        logistics = order_info.logistics[0]  # 一个订单，只有一个物流，返回的是 list，直接取 0 即可
        message = [f"- **订单ID**: {order_id}"]
        message.extend(
            [
                f"- {order_detail.sku_name} × {order_detail.sku_count}"
                for order_detail in order_info.order_detail
            ]
        )
        message.append(f"- **物流ID**: {logistics.logistics_id}")
        message.append("- **物流信息**:")
        message.append(
            " - "
            + "\n - ".join(
                logistics.logistics_tracking.split("\n")
            )
        )

        #4、返回给用户
        dispatcher.utter_message("\n".join(message))

        return [SlotSet("logistics_id", logistics.logistics_id)]

class GetLogisticsComplaintList(Action):
    """
    查询数据库中常见投诉原因，返回button
    """

    def name(self) -> Text:
        return "action_ask_complaint_content"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # 1、查询指定物流id对应的物流信息
        logistics_id = tracker.get_slot("logistics_id")

        with session_local() as session:
            logistics = (
                session.query(Logistics)
                .filter(Logistics.logistics_id == logistics_id)
                .first()
            )

        # 2、判断是已发货还是已签收
        logistics_status="已发货" if logistics.delivered_time is None else "已签收"
        # 3、查询数据库logistics_complaint表,获取常见投诉原因
        with session_local() as session:
            logistics_complaint = (
                session.query(LogisticsComplaint)
                .filter(LogisticsComplaint.logistics_status == logistics_status)
                .all()
            )

        # 4、封装返回消息
        buttons = [
            {
                "title": complaint.logistics_complaint,
                "payload": f"/SetSlots(complaint_content={complaint.logistics_complaint})"
            }
            for complaint in logistics_complaint
        ]

        # 5、添加其它、取消投诉的按钮
        buttons.append(
            {
                "title": "其它",
                "payload": "/SetSlots(complaint_content=other)"
            })
        buttons.append(
            {
                "title": "取消投诉",
                "payload": "/SetSlots(complaint_content=false)"
            })

        # 5、返回给用户
        dispatcher.utter_message(text="请选择要反馈的问题：", buttons=buttons)

        return []

class InsertLogisticsComplaint(Action):
    """
    记录物流投诉到数据库
    """

    def name(self) -> Text:
        return "action_record_logistics_complaint"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1、获取数据
        logistics_id = tracker.get_slot("logistics_id")
        complaint_content = tracker.get_slot("complaint_content")

        # 初始化等级
        complaint_level = None

        # 2、判断是否 other
        if complaint_content == "other":
            complaint_content = tracker.get_slot("other_logistics_complaint")

            # 模型预测
            complaint_level = self.predict_level(complaint_content)

        else:
            # 按钮映射
            complaint_level = BUTTON_LEVEL_MAP.get(complaint_content, "low")

        # 3、插入数据库（关键改动）
        with session_local() as session:
            session.add(
                LogisticsComplaintsRecord(
                    logistics_id=logistics_id,
                    logistics_complaint=complaint_content,
                    complaint_level=complaint_level,
                    complaint_time=datetime.now(),
                    user_id=tracker.get_slot("user_id")
                )
            )
            session.commit()

        # 4、返回给用户
        dispatcher.utter_message(
            text=f"已收到您的投诉，投诉等级为：{complaint_level}"
        )

        return []

    def predict_level(self,text: str) -> str:
        tokens = tokenizer(
            text,
            max_length=85,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        pred_id = int(torch.argmax(logits, dim=1).item())
        return ["low", "medium", "high"][pred_id]

