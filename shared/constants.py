import string
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent
GT_PATH    = ROOT_DIR / "data" / "ground_truth_absa.csv"
DATA_PATH  = ROOT_DIR / "data" / "cleaned_texts.csv"
CACHE_DIR  = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"

CATEGORIES = ["UI_UX", "ANIMATION", "CONTENT", "QUIZ", "PERFORMANCE", "GAMIFICATION", "OTHERS"]

# ── POS ──────────────────────────────────────────────────────────────────────

CKIP_ASPECT_POS    = frozenset({"Na", "Nb", "Nv", "FW"})
CKIP_OPINION_POS   = frozenset({"VH", "VJ", "A"})
# Broader noun POS used in OTE/AOPE context checks (includes Nf)
CKIP_NOUN_POS      = frozenset({"Na", "Nb", "Nv", "Nf", "FW"})
_MODIFIER_NOUN_POS = frozenset({"Na", "Nb", "Nv", "Nf"})

_PUNCT = frozenset(string.punctuation + "，。！？、；：“”‘’（）【】「」『』…—～·")

# ── Lexical ───────────────────────────────────────────────────────────────────

DEGREE_ADV = frozenset({
    "很", "太", "非常", "相當", "極", "超", "蠻", "頗", "最",
    "更", "比較", "還", "真", "挺", "有點", "一點點", "超級",
})
STOP_WORDS = frozenset({"我", "你", "他", "她", "它", "這", "那", "什麼", "時候"})
WEAK_ASPECT_NOUNS = frozenset({
    "之類", "部分", "地方", "方面", "事情", "東西", "情況",
    "方式", "過程", "程度", "附近",
})
NEUTRAL_OPINION_WORDS = frozenset({
    "呈現", "對照", "並重", "保有", "送出", "提高", "更新", "追蹤",
    "加上", "提醒", "驗證", "推出", "融為",
    "知道", "發現", "理解", "學習", "明白", "懂得",
    "看", "懂", "來", "去", "做", "用", "有", "無",
    "時", "沒有", "這樣", "不同", "快", "生成", "自有", "高度", "對",
    "多", "細", "久", "即時", "好",
})

VHC_WHITELIST        = frozenset({"視覺化", "可視化", "美化", "簡化", "最佳化", "豐富化", "多元化"})
CUSTOM_OPINION_WORDS = frozenset({"牛逼", "卡", "掛", "直觀", "便利"})
CUSTOM_ASPECT_WORDS  = frozenset({"提交", "講解", "提示", "指引"})

NEGATION_PREFIXES     = frozenset({"不", "沒", "非", "未", "不太", "不會", "不夠", "不算", "不很", "沒有"})
NEGATION_INTENSIFIERS = frozenset({"太", "很", "一點", "一點點", "有點", "稍微", "稍", "十分"})
NEG_BASE_OPINIONS     = frozenset({
    "複雜", "繁瑣", "難", "難懂", "難理解", "亂", "錯亂",
    "差", "爛", "慢", "麻煩", "混亂",
})
NEG_COGNITIVE_VERBS  = frozenset({"知道", "理解", "明白", "懂得"})
PC_VERB_POS          = frozenset({"VJ", "VH", "VC", "VE", "VK"})

FIRST_PERSON_PRONOUNS   = frozenset({"我", "我們", "自己", "我自己"})
COGNITIVE_VERBS_EXCLUDE = frozenset({
    "知道", "理解", "發現", "學習", "明白", "懂得",
    "看", "感到", "覺得", "認為", "想", "看到", "注意",
})

# ── V2 rule ───────────────────────────────────────────────────────────────────

V2_POSITIVE_TOKENS = frozenset({"有"})
V2_POSITIVE_POS    = frozenset({"V_2"})
V2_NEGATIVE_TOKENS = frozenset({"沒有", "沒"})
MODAL_BEFORE_V2    = frozenset({"會", "可能", "應該", "必須", "要", "想", "可以"})
NOUN_COLLECT_POS   = frozenset({"Na", "Nb", "Nv", "Nf"})
NOUN_SKIP_POS      = frozenset({"DE"})
MIN_ASPECT_LEN     = 2

# ── Compound noun merging (A3) ────────────────────────────────────────────────

A3_COMPOUND_V = frozenset({"VC", "VE", "VD"})
A3_OPINION_POS = frozenset({"VH", "VJ", "VK", "A"})

# ── Category seeds ────────────────────────────────────────────────────────────

CATEGORY_SEEDS = {
    "UI_UX":        ["介面", "畫面", "按鈕", "顏色", "排版", "手機版", "舒適", "乾淨", "單調", "設計",
                     "好看", "外觀", "美觀", "版面", "導航", "頁面", "佈局",
                     "操作", "指引", "功能", "提示", "觀感", "圖表"],
    "ANIMATION":    ["動畫", "演示", "逐步", "逐行", "視覺化", "步驟", "播放", "動態", "逐格",
                     "動態演示", "圖示化", "步步"],
    "CONTENT":      ["中文", "翻譯", "解釋", "程式碼", "涵義", "概念", "虛擬碼", "摘要",
                     "時間複雜度", "教學", "講解", "課程", "說明"],
    "QUIZ":         ["測驗", "題目", "難度", "答題", "解析", "錯誤檢測", "作答",
                     "考卷", "成績", "分數", "答案"],
    "PERFORMANCE":  ["卡頓", "等待", "延遲", "Bug", "失敗", "提交", "紀錄", "即時", "卡",
                     "同步", "錯誤", "回應", "當掉"],
    "GAMIFICATION": ["遊戲", "互動", "彩蛋", "好玩", "闖關", "挑戰", "沙箱", "玩", "幽默",
                     "排行", "打地鼠", "排行榜", "積分", "成就"],
}

# A2-specific seeds (fewer, removes opinion-overlap words)
CATEGORY_SEEDS_A2 = {
    "UI_UX":        ["介面", "畫面", "按鈕", "設計", "外觀", "版面", "導航", "頁面"],
    "ANIMATION":    ["動畫", "演示", "視覺化", "播放", "動態", "動畫演示"],
    "CONTENT":      ["程式碼", "虛擬碼", "摘要", "時間複雜度", "教學", "講解", "課程"],
    "QUIZ":         ["測驗", "題目", "解析", "錯誤檢測", "作答", "詳解", "考卷"],
    "PERFORMANCE":  ["卡頓", "失敗", "提交", "紀錄", "延遲", "回應"],
    "GAMIFICATION": ["遊戲", "彩蛋", "沙箱", "打地鼠", "挑戰", "積分"],
}

# ── Model names ───────────────────────────────────────────────────────────────

SBERT_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
DISTILBERT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
