# Day 14 — Reflection

## 1. Failure Analysis (5 Whys)

*Phân tích 3 trường hợp bị đánh giá kém nhất từ kết quả Benchmark (xem Exercises 3.2)*

### Failure 1: ID M06 (Score: 0.42)
- **Question:** Đơn hàng báo đã giao nhưng tôi chưa nhận được. Tôi phải làm gì?
- **Actual Agent Answer:** Chúng tôi đền bù 1.000.000đ cho mỗi đơn hàng bị thất lạc.
- **Expected Answer:** Hãy kiểm tra với hàng xóm trước, sau đó liên hệ CSKH trong vòng 48h.
- **Failure Type:** Hallucination
- **5 Whys Analysis:**
  1. **Why did the agent hallucinate?** Bởi vì agent không tìm thấy câu trả lời trực tiếp trong retrieved chunks và bắt đầu tự sinh (hallucinate) ra quy định đền bù tiền.
  2. **Why didn't the agent find the answer in retrieved chunks?** Bởi vì retrieved chunks bị cắt quá ngắn (chunk fragmentation) và bỏ sót tài liệu "Thất lạc hàng" trong top k.
  3. **Why was the relevant document missed?** Vì query của user có cách diễn đạt hơi khác với văn bản gốc nên BM25 retrieval (lexical search) bị miss từ khoá quan trọng.
  4. **Why wasn't semantic search compensating?** Semantic search có kéo tài liệu lên nhưng bị rớt khỏi top-k do nhiễu (noise) từ các chunks quảng cáo sản phẩm.
  5. **Why was there noise in top-k?** Do chưa có Reranking step (Context Precision thấp) để đưa chunk chính xác lên đầu.
- **Root Cause:** Thiếu Reranking step và giới hạn Top-k quá hẹp, kết hợp với Faithfulness Guardrail yếu.

### Failure 2: ID H03 (Score: 0.15)
- **Question:** Làm sao để đổi quà tặng mà người tặng không biết?
- **Actual Agent Answer:** Bạn phải xin người tặng biên lai, sau đó chúng tôi sẽ gửi email xác nhận cho họ.
- **Expected Answer:** Yêu cầu đổi bằng biên lai quà tặng để nhận thẻ tích điểm. Người tặng sẽ không bị thông báo.
- **Failure Type:** Hallucination / Refusal
- **5 Whys Analysis:**
  1. **Why did the agent say the sender will be notified?** Bởi vì agent áp dụng nhầm chính sách "Trả hàng" tiêu chuẩn (luôn gửi email cho người mua) thay vì chính sách "Quà tặng".
  2. **Why did it apply standard Returns policy?** Vì từ khoá "đổi" kích hoạt chunk của "Chính sách trả hàng" mạnh hơn là "Quà tặng".
  3. **Why didn't it use the Gift Returns chunk?** Chunk "Quà tặng" nằm ở vị trí thứ 6 trong retrieved context, và LLM bị dính Position Bias (chỉ chú ý thông tin đầu) dẫn đến bỏ qua chunk thứ 6.
  4. **Why was the chunk at position 6?** Retriever không hiểu sâu sắc được semantics của "mà người tặng không biết" để map với "không bị thông báo", dẫn đến ranking sai.
  5. **Why did it confidently answer wrong?** Do không có guardrail nhắc LLM "nếu không chắc chắn hoặc context mâu thuẫn, hãy yêu cầu làm rõ".
- **Root Cause:** LLM Position Bias (lost in the middle) và thiếu Semantic Reranking.

### Failure 3: ID M02 (Score: 0.65)
- **Question:** Giao hàng tiêu chuẩn tới Phú Quốc mất bao lâu?
- **Actual Agent Answer:** Giao hàng tiêu chuẩn mất 3-5 ngày.
- **Expected Answer:** Giao hàng tới Phú Quốc mất 5-7 ngày làm việc.
- **Failure Type:** Incomplete
- **5 Whys Analysis:**
  1. **Why is the answer incomplete?** Agent chỉ nêu thời gian vận chuyển nội địa chung (3-5 ngày) mà bỏ sót ngoại lệ cho Phú Quốc (5-7 ngày).
  2. **Why was the exception missed?** Câu hỏi chứa chữ "Phú Quốc", nhưng chunk quy định bị tách biệt làm 2 câu: câu 1 nói về nội địa chung, câu 2 nói về huyện đảo/Phú Quốc.
  3. **Why did the LLM only read the first sentence?** Do câu lệnh (prompt) thiết kế chung chung, không ép LLM phải check exception (các trường hợp ngoại lệ).
  4. **Why wasn't the prompt enforcing exception checks?** Prompt hiện tại chỉ yêu cầu "Answer the question based on the context", không yêu cầu "Think step-by-step to identify special conditions".
  5. **Why did Completeness metric drop?** Vì answer không phủ được từ khoá "5-7 ngày làm việc".
- **Root Cause:** Prompt thiếu kỹ thuật Chain-of-Thought (CoT) để LLM cẩn thận rà soát các điều kiện ngoại lệ trong policies.

---

## 2. Improvement Log

Dựa trên phân tích 5 Whys, dưới đây là danh sách các hành động ưu tiên:

| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| M06 | Hallucination | Context Precision kém do nhiễu, thiếu reranker | Cài đặt Cohere Rerank / BGE-Reranker sau bước lấy 20 chunks để giữ lại top 5 chunk liên quan nhất. | Open |
| H03 | Hallucination | Lost in the middle (Position Bias) | Giảm top-k context window hoặc thêm hướng dẫn "đọc toàn bộ context từ đầu đến cuối trước khi trả lời" vào system prompt. | Open |
| M02 | Incomplete | Thiếu Chain-of-Thought cho các quy định ngoại lệ | Cập nhật prompt: "Identify any exceptions in the policy before writing the final answer." | Open |
| A01-A03 | Out-of-scope | Thiếu Input Guardrail | Bổ sung lớp phân loại Intent bằng NLP hoặc LLM nhẹ để chặn các câu lệnh tiêm nhiễm (Prompt Injection). | Open |

---

## 3. Chiến lược CI/CD Integration

Để Evaluation Pipeline trở thành Quality Gate thực sự, tôi sẽ tích hợp script này vào **GitHub Actions** (hoặc GitLab CI).

**Workflow Pipeline:**
1. **Trigger:** `On Pull Request` (khi Dev sửa prompt, update agent logic hoặc thay đổi RAG chunking parameters).
2. **Execute:** 
   - Kéo file `test_eval.py` và `BenchmarkRunner`.
   - Lấy 20 câu hỏi từ `Golden Dataset` (E-commerce).
   - Chạy mô phỏng agent với test queries.
3. **Assert Thresholds (Quality Gates):**
   - Fail pipeline nếu `Avg Faithfulness < 0.85`.
   - Fail pipeline nếu `Avg Answer Relevancy < 0.70`.
   - Fail pipeline nếu `Avg Context Precision < 0.60`.
4. **Regression Check:**
   - Script gọi API lấy điểm số của nhánh `main` (Baseline).
   - Chạy hàm `run_regression()`, nếu bất kỳ metric nào bị drop (giảm) > 5% so với Baseline -> Đánh dấu Pull Request là **FAILED**.
5. **Report Generation:**
   - Tự động xuất kết quả Benchmark và Failure Analysis Log (dạng Markdown) vào comment của Pull Request để Reviewer kiểm tra.
   - Các cases failed sẽ được log lại thành Jira tickets để debug.
