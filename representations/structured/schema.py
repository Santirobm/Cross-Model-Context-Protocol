"""Structured CMCP context schema and compact wire-prompt rendering."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    kind: str = "object"
    value: str = ""


class Relation(BaseModel):
    src: str
    rel: str
    dst: str


class Instruction(BaseModel):
    text: str
    priority: str = "normal"


class Constraint(BaseModel):
    rule: str
    scope: str = "global"


class Evidence(BaseModel):
    id: str
    claim: str
    source: str = ""


class Reference(BaseModel):
    id: str
    uri: str
    digest: str = ""
    resolved_text: str = ""


class Decision(BaseModel):
    topic: str
    choice: str
    rationale: str = ""


class CMCPContext(BaseModel):
    """Canonical structured context for cross-model transfer."""

    task_id: str = ""
    concepts: List[str] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    instructions: List[Instruction] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    evidences: List[Evidence] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)

    def to_json_str(self, compact: bool = True) -> str:
        """Render as canonical JSON string."""
        if compact:
            return self.model_dump_json(exclude_defaults=True)
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json_str(cls, data: str) -> "CMCPContext":
        """Parse from JSON string."""
        return cls.model_validate_json(data)

    def to_compact_text(self, include_resolved_refs: bool = False) -> str:
        """Render as token-efficient human-readable CMCP structured lines."""
        lines: List[str] = []
        if self.task_id:
            lines.append(f"TASK:{self.task_id}")
        if self.concepts:
            lines.append("CONCEPT:" + "|".join(self.concepts))
        for e in self.entities:
            val = f"={e.value}" if e.value else ""
            lines.append(f"ENT:{e.name}({e.kind}){val}")
        for r in self.relations:
            lines.append(f"REL:{r.src}->{r.rel}->{r.dst}")
        for ins in self.instructions:
            prio = f"[{ins.priority}]" if ins.priority != "normal" else ""
            lines.append(f"INS{prio}:{ins.text}")
        for c in self.constraints:
            sc = f"[{c.scope}]" if c.scope and c.scope != "global" else ""
            lines.append(f"CON{sc}:{c.rule}")
        for ev in self.evidences:
            src = f" @{ev.source}" if ev.source else ""
            lines.append(f"EVD[{ev.id}]:{ev.claim}{src}")
        for ref in self.references:
            if include_resolved_refs and ref.resolved_text:
                lines.append(f"REF[{ref.id}]:{ref.resolved_text}")
            else:
                dig = f"#{ref.digest}" if ref.digest else ""
                lines.append(f"REF[{ref.id}]:{ref.uri}{dig}")
        for dec in self.decisions:
            rat = f" ({dec.rationale})" if dec.rationale else ""
            lines.append(f"DEC[{dec.topic}]={dec.choice}{rat}")
        return "\n".join(lines)

    def to_wire_prompt(
        self,
        task_query: str = "",
        include_resolved_refs: bool = False,
    ) -> str:
        """Render a concise wire prompt for a receiver LLM."""
        ctx_block = self.to_compact_text(include_resolved_refs=include_resolved_refs)
        if task_query:
            return (
                "CMCP_CTX:\n"
                f"{ctx_block}\n"
                f"QUERY: {task_query}\n"
                "Answer concisely using only facts, constraints, decisions, and evidence in CMCP_CTX."
            )
        return f"CMCP_CTX:\n{ctx_block}"

    def to_tlv_records(self) -> List[Tuple[int, bytes]]:
        """Convert structured fields into (tag, utf8_bytes) pairs for binary TLV encoding."""
        records: List[Tuple[int, bytes]] = []
        # Tag constants match core/include/cmcp/cmcp.h
        for c in self.concepts:
            records.append((0x01, c.encode("utf-8")))
        for e in self.entities:
            records.append((0x02, e.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for r in self.relations:
            records.append((0x03, r.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for ins in self.instructions:
            records.append((0x04, ins.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for con in self.constraints:
            records.append((0x05, con.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for ev in self.evidences:
            records.append((0x06, ev.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for ref in self.references:
            records.append((0x07, ref.model_dump_json(exclude_defaults=True).encode("utf-8")))
        for dec in self.decisions:
            records.append((0x08, dec.model_dump_json(exclude_defaults=True).encode("utf-8")))
        return records

    @classmethod
    def from_tlv_records(
        cls,
        records: List[Tuple[int, bytes]],
        task_id: str = "",
    ) -> "CMCPContext":
        """Reconstruct a CMCPContext from decoded TLV (tag, value_bytes) pairs."""
        ctx = cls(task_id=task_id)
        for tag, raw in records:
            text = raw.decode("utf-8")
            if tag == 0x01:
                ctx.concepts.append(text)
            elif tag == 0x02:
                ctx.entities.append(Entity.model_validate_json(text))
            elif tag == 0x03:
                ctx.relations.append(Relation.model_validate_json(text))
            elif tag == 0x04:
                ctx.instructions.append(Instruction.model_validate_json(text))
            elif tag == 0x05:
                ctx.constraints.append(Constraint.model_validate_json(text))
            elif tag == 0x06:
                ctx.evidences.append(Evidence.model_validate_json(text))
            elif tag == 0x07:
                ctx.references.append(Reference.model_validate_json(text))
            elif tag == 0x08:
                ctx.decisions.append(Decision.model_validate_json(text))
        return ctx
