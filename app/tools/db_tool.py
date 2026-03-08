"""
DB Tool：Structured DB retrieval for NIH stage guidance/definitions
"""
import re
from app.tools.base import BaseTool
from app.core.types import ToolResult, Citation


class DBTool(BaseTool):
    """Database retrieval tool"""
    
    # Mock database (replace with real DB in production)
    STAGE_DEFINITIONS = {
        "0": {
            "name": "Stage 0",
            "description": "basic researchstage：interventionmechanism",
            "requirements": ["basic research", "mechanism", "hypothesis generation"],
            "next_stage": "Stage I"
        },
        "I": {
            "name": "Stage I",
            "description": "Feasibility research: manualization and pilot testing.",
            "requirements": ["intervention", "manualization", "small-sample feasibility study"],
            "next_stage": "Stage II"
        },
        "II": {
            "name": "Stage II",
            "description": "efficacy：mechanismrandomized controlled trial",
            "requirements": ["mechanism", "randomized controlled trial", "efficacy"],
            "next_stage": "Stage III"
        },
        "III": {
            "name": "Stage III",
            "description": "effectiveness：",
            "requirements": ["real-world testing", "diverse samples", "effectiveness"],
            "next_stage": "Stage IV"
        },
        "IV": {
            "name": "Stage IV",
            "description": "implementation：disseminationscale-up",
            "requirements": ["implementation", "dissemination", "scale-up"],
            "next_stage": "Stage V"
        },
        "V": {
            "name": "Stage V",
            "description": "Sustainability research: long-term maintenance and optimization.",
            "requirements": ["sustainability", "long-term maintenance", "continuous optimization"],
            "next_stage": None
        }
    }
    
    def __init__(self):
        super().__init__("db_tool", "Structured retrieval for NIH stage guidance")
    
    def run(self, query: str, stage: str = None, **kwargs) -> ToolResult:
        """Run database query"""
        try:
            query_lower = query.lower()

            # If stage is explicitly provided, return directly
            if stage and stage in self.STAGE_DEFINITIONS:
                data = self.STAGE_DEFINITIONS[stage]
                return ToolResult(
                    text=f"Stage {stage}: {data['description']}\nrequirements: {', '.join(data['requirements'])}",
                    structured=data,
                    citations=[
                        Citation(
                            source="NIH Stage Model Database",
                            passage=data['description'],
                            relevance_score=1.0
                        )
                    ],
                    success=True
                )

            # If query explicitly contains stage token
            explicit_match = re.search(r"stage\s*(0|i{1,3}|iv|v)\b", query_lower, flags=re.IGNORECASE)
            if explicit_match:
                token = explicit_match.group(1).upper()
                if token in self.STAGE_DEFINITIONS:
                    data = self.STAGE_DEFINITIONS[token]
                    return ToolResult(
                        text=f"{data['name']}: {data['description']}\nrequirements: {', '.join(data['requirements'])}",
                        structured=data,
                        citations=[
                            Citation(
                                source="NIH Stage Model Database",
                                passage=data["description"],
                                relevance_score=0.98,
                            )
                        ],
                        success=True,
                    )

            # NIH Stage Model definition query
            if "nih stage model" in query_lower or "what is" in query_lower:
                overview = []
                for key in ["0", "I", "II", "III", "IV", "V"]:
                    d = self.STAGE_DEFINITIONS[key]
                    overview.append(f"{d['name']}: {d['description']}")
                overview_text = "NIH Stage Model overview:\n" + "\n".join(overview)
                return ToolResult(
                    text=overview_text,
                    structured={"overview": self.STAGE_DEFINITIONS},
                    citations=[
                        Citation(
                            source="NIH Stage Model Database",
                            passage="NIH Stage Model intervention Stage 0  Stage V。",
                            relevance_score=0.95,
                        )
                    ],
                    success=True,
                )

            # Otherwise search matching stage entries
            results = []

            for stage_key, stage_data in self.STAGE_DEFINITIONS.items():
                req_hit = any(req.lower() in query_lower for req in stage_data["requirements"])
                desc_hit = any(tok in query_lower for tok in stage_data["description"].lower().split())
                if req_hit or desc_hit:
                    results.append({
                        "stage": stage_key,
                        **stage_data
                    })
            
            if results:
                # Return first matched result
                best_match = results[0]
                return ToolResult(
                    text=f"Matched: {best_match['name']}: {best_match['description']}",
                    structured={"matches": results, "best_match": best_match},
                    citations=[
                        Citation(
                            source="NIH Stage Model Database",
                            passage=best_match['description'],
                            relevance_score=0.9
                        )
                    ],
                    success=True
                )
            else:
                return ToolResult(
                    text="No matching result found stage ",
                    structured={"matches": []},
                    success=True
                )
        
        except Exception as e:
            return ToolResult(
                text=f"Query error: {str(e)}",
                success=False,
                error=str(e)
            )
