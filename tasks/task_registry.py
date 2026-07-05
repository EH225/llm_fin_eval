"""
Task Registry - This module define 6 typical financial analysis tasks across 3 difficulty tiers.

Each task includes:
    - A realistic prompt referencing public filings and/or market data
    - The ground truth answer(s) expected
    - Grader type and parameters
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    id: str
    name: str
    tier: int  # Difficulty setting 1=Easy, 2=Medium, 3=Hard
    prompt: str
    context: str  # Data the agent can "retrieve" via tool
    ground_truth: Any
    grader_type: str  # "exact", "tolerance", "range", "llm_rubric"
    grader_params: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)


# Define a set of tasks to be evaluated by
TASKS = [

    ###                                      ###
    ### Tier 1: Data extraction tasks (easy) ###
    ###                                      ###
    Task(
        id="T1",
        name="EPS Extraction - Apple 10-K FY2025",
        tier=1,
        prompt=(
            "Using the Apple Inc. 10-K filing for fiscal year 2025 (year ended "
            "September 30, 2025), what was the diluted earnings per share (EPS)? "
            "Report the number exactly as stated in the filing. "
            "See below for the relevant data extracted for you to reference."
        ),
        context="""
Apple Inc. Consolidated Statements of Operations (in millions, except per share amounts)
Fiscal Year Ended September 30, 2025

Net sales:
  Products                        $307,003
  Services                        $109,158
Total net sales                   $416,161

Cost of sales:
  Products                        $194,116
  Services                         $26,844
Total cost of sales               $220,960

Gross margin                      $195,201

Operating expenses:
  Research and development         $34,550
  Selling, general and admin       $27,601
Total operating expenses           $62,151

Operating income                  $133,050
Other income/(expense), net          $(321)
Income before provision            $132,729
Provision for income taxes          $20,719
Net income                          $112,010

Earnings per share:
  Basic                               $7.49
  Diluted                             $7.46

Shares used in computing EPS (thousands):
  Basic                          14,948,500
  Diluted                        15,004,697
""",
        ground_truth="$7.46",
        grader_type="exact",
        grader_params={"aliases": ["7.46", "$7.46", "7.46 per share"]},
        tags=["extraction", "10-K", "EPS"],
    ),

    Task(
        id="T2",
        name="Revenue CAGR: Microsoft FY2021 - FY2023",
        tier=1,
        prompt=(
            "Using Microsoft's annual revenue figures for fiscal years 2021, 2022, "
            "and 2023, calculate the 2-year compound annual growth rate (CAGR) of "
            "total revenue from FY2021 to FY2023. Express your answer as a percentage "
            "rounded to one decimal place (e.g. '15.3%'). "
            "See below for the relevant data extracted for you to reference."
            "Please only provide 1 answer, your primary answer."
        ),
        context="""
Microsoft Corporation - Revenue Summary (USD millions)

Fiscal Year 2021 (ended June 30, 2021):
  Productivity and Business Processes:  $53,915
  Intelligent Cloud:                    $60,080
  More Personal Computing:              $54,093
  Total Revenue:                       $168,088

Fiscal Year 2022 (ended June 30, 2022):
  Productivity and Business Processes:  $63,364
  Intelligent Cloud:                    $75,251
  More Personal Computing:              $59,655
  Total Revenue:                       $198,270

Fiscal Year 2023 (ended June 30, 2023):
  Productivity and Business Processes:  $69,274
  Intelligent Cloud:                    $87,907
  More Personal Computing:              $54,734
  Total Revenue:                       $211,915
""",
        # CAGR = (211915/168088)^(1/2) - 1 = 0.1225 = 12.3% (rounded to 1 decimal after % conversion)
        ground_truth=12.3,
        grader_type="tolerance",
        grader_params={"tolerance_pct": 0.5, "unit": "percent"},
        tags=["calculation", "CAGR", "revenue"],
    ),

    ###                                        ###
    ### Tier 2: Data extraction tasks (medium) ###
    ###                                        ###
    Task(
        id="T3",
        name="Debt-to-Equity Ratio - Meta Platforms Q4 2023",
        tier=2,
        prompt=(
            "Using Meta Platforms' balance sheet as of December 31, 2023, "
            "calculate the debt-to-equity ratio. Round to three decimal places. "
            "See below for the relevant data extracted for you to reference."
            "Please only provide 1 answer, your primary answer."
        ),
        context="""
Meta Platforms, Inc. - Consolidated Balance Sheets (in millions)
As of December 31, 2023

ASSETS
Current assets:
  Cash and cash equivalents               $41,862
  Marketable securities                   $23,541
  Accounts receivable, net                $19,013
  Prepaid expenses and other               $5,765
Total current assets                      $90,181

Non-current assets:
  Property and equipment, net             $96,587
  Operating lease right-of-use assets     $14,217
  Intangible assets, net                   $1,036
  Goodwill                                $20,654
  Other assets                             $8,054
Total assets                             $229,623 (note: sum may include rounding)

LIABILITIES AND STOCKHOLDERS' EQUITY
Current liabilities:
  Accounts payable                         $4,849
  Partners payable                         $1,169
  Operating lease liabilities, current     $1,626
  Accrued expenses and other              $22,212
  Deferred revenue                           $519
Total current liabilities                 $30,375 (note: sum may include rounding)

Non-current liabilities:
  Operating lease liabilities, non-current $17,621
  Long-term debt                            $8,455 (net of discount/issuance costs)
  Other liabilities                        $11,013
Total liabilities                          $67,364 (note: sum may include rounding; check for latest filing)

Stockholders' equity:
  Common stock and additional paid-in capital  $73,253
  Accumulated other comprehensive loss         ($2,983)
  Retained earnings                            $91,989
Total stockholders' equity                    $153,168 (note: as reported)

Note: figures are as reported; minor rounding differences may appear between summed line items and totals.
""",
        # D/E = 8,455 / 153,168 = 0.055
        ground_truth=0.055,
        grader_type="tolerance",
        grader_params={"tolerance_pct": 5.0, "unit": "ratio"},
        tags=["calculation", "balance-sheet", "leverage"],
    ),

    Task(
        id="T4",
        name="Bond duration calculation",
        tier=2,
        prompt=(
            "If a bond has a price of $101, a face value of $100, an interest rate of"
            "5.5%, annual coupons, and a time-ot-maturity of 5 years, what is its modified duration?"
            "Please only provide 1 answer, your primary answer, rounded to 3 decimal places."
        ),
        context="",
        ground_truth=4.282,
        grader_type="tolerance",
        grader_params={"tolerance_pct": 1.0, "unit": "ratio"},
        tags=["calculation", "bond-math", "duration"],
    ),

    ###                                       ###
    ### Tier 3: Analysis and Synthesis (hard) ###
    ###                                       ###
    Task(
        id="T5",
        name="Free Cash Flow Quality Analysis - Alphabet FY2023",
        tier=3,
        prompt=(
            "Based on Alphabet's FY2023 cash flow statement, write a 150-250 word "
            "analysis of the *quality* of its free cash flow. Your analysis should: "
            "(1) calculate FCF (operating cash flow minus capex), "
            "(2) compare FCF to net income and comment on accrual quality, "
            "(3) assess whether capex is growth-oriented or maintenance, and "
            "(4) identify any one-time items or unusual patterns. "
            "See below for the relevant data extracted for you to reference."
        ),
        context="""
Alphabet Inc. - Consolidated Statements of Cash Flows (in millions)
Year Ended December 31, 2023

OPERATING ACTIVITIES
  Net income                                              $73,795
  Depreciation and amortization                           $11,946
  Stock-based compensation expense                        $22,460
  Deferred income taxes                                  ($7,380)
  Loss on debt instruments, net                              $593
  Other                                                    $3,099
  Changes in working capital:
    Accounts receivable                                  ($2,529)
    Income taxes                                          $3,337
    Other assets                                         ($4,561)
    Accounts payable                                      $1,497
    Accrued expenses and other liabilities                $7,314
Net cash from operating activities                      $101,746 (note: check totals)

INVESTING ACTIVITIES
  Capital expenditures                                  ($32,251)
  Purchases of marketable securities                    ($67,713)
  Proceeds from maturities of marketable securities      $46,402
  Proceeds from sales of marketable securities           $37,056
  Acquisitions, net of cash                              ($2,545)
  Other investing activities                              $1,589
Net cash used in investing activities                   ($24,055) (note: reflects netting)

FINANCING ACTIVITIES
  Net share repurchases                                  ($61,504)
  Proceeds from issuance of debt                             $997
  Repayments of debt                                     ($1,000)
  Other financing activities                             ($1,045)
Net cash used in financing activities                   ($62,552)

Free Cash Flow (as defined by Alphabet):
  Operating cash flow                                   $101,746 (note: check totals)
  Less: Capital expenditures                            ($32,251)
  Free Cash Flow                                         $69,495 (note: check totals)
""",
        ground_truth={
            "fcf_usd_billions": 69.5,
            "key_themes": [
                "FCF significantly exceeds net income due to high D&A and SBC",
                "FCF conversion ratio well above 90%, indicating strong cash quality",
                "Capex of $32B is heavily growth-oriented (data centers, AI infrastructure)",
                "SBC of $22.5B is large non-cash charge; adds back to OCF but dilutes shareholders",
                "No material one-time items; operating pattern is clean",
            ],
        },
        grader_type="llm_rubric",
        grader_params={
            "rubric": [
                {"criterion": "FCF calculated correctly (~$69-70B)", "weight": 20},
                {
                    "criterion": "FCF vs net income comparison made with correct directional insight (FCF > NI due to D&A+SBC)",
                    "weight": 25},
                {
                    "criterion": "SBC identified as significant non-cash add-back with shareholder dilution caveat",
                    "weight": 20},
                {
                    "criterion": "Capex characterized as growth/AI-infrastructure oriented, not pure maintenance",
                    "weight": 20},
                {"criterion": "Writing is concise, precise, and within word count", "weight": 15},
            ],
            "max_score": 100,
            "pass_threshold": 85,
        },
        tags=["analysis", "cash-flow", "quality"],
    ),

    Task(
        id="T6",
        name="Gross Margin Comparison - FAANG 2023",
        tier=3,
        prompt=(
            "Using FY2023 income statements for Apple, Microsoft, Alphabet, and Meta, "
            "rank these companies by gross margin percentage (highest to lowest) and "
            "provide a 100-150 word explanation of what drives the differences. "
            "Your ranking must be explicitly stated as a numbered list before the explanation. "
            "See below for the relevant data extracted for you to reference."
        ),
        context="""
Apple FY2023: Total Revenue $383,285M | Gross Profit $169,148M

Microsoft FY2023: Total Revenue $211,915M | Gross Profit $146,052M

Alphabet FY2023: Total Revenue $307,394M | Gross Profit $174,062M (note: represents gross profit after traffic acquisition costs and content costs)

Meta FY2023: Total Revenue $134,902M | Gross Profit $115,807M (note: represents revenue minus cost of revenue)
""",
        ground_truth={
            "ranking": ["Meta", "Microsoft", "Alphabet", "Apple"],
            "margins": {"Meta": 85.8, "Microsoft": 68.9, "Alphabet": 56.6, "Apple": 44.1},
        },
        grader_type="llm_rubric",
        grader_params={
            "rubric": [
                {"criterion": "Ranking is exactly correct: Meta > Microsoft > Alphabet > Apple",
                 "weight": 40},
                {"criterion": "Margin percentages are approximately correct (within 1pp) for all four",
                 "weight": 20},
                {"criterion": "Explanation addresses Meta's software-only cost structure", "weight": 15},
                {"criterion": "Explanation addresses Apple's hardware drag on margins", "weight": 15},
                {"criterion": "Explanation is within word count and analytically coherent", "weight": 10},
            ],
            "max_score": 100,
            "pass_threshold": 85,
        },
        tags=["synthesis", "comparison", "margins"],
    ),
]

TASK_MAP = {t.id: t for t in TASKS}
