#include "excel_export.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace {
    std::string escape_xml(const std::string& data) {
        std::string buffer;
        buffer.reserve(data.size() * 1.1);
        for (char c : data) {
            switch (c) {
                case '&':  buffer.append("&amp;");       break;
                case '\"': buffer.append("&quot;");      break;
                case '\'': buffer.append("&apos;");      break;
                case '<':  buffer.append("&lt;");        break;
                case '>':  buffer.append("&gt;");        break;
                default:   buffer.push_back(c);          break;
            }
        }
        return buffer;
    }
}

bool ExcelExporter::exportToExcel(const std::string& filename, 
                                 const std::vector<BenchmarkRecord>& records,
                                 const HardwareInfo& hw) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[Lỗi] Không thể tạo file Excel: " << filename << std::endl;
        return false;
    }

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<?mso-application progid=\"Excel.Sheet\"?>\n";
    out << "<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\"\n";
    out << " xmlns:o=\"urn:schemas-microsoft-com:office:office\"\n";
    out << " xmlns:x=\"urn:schemas-microsoft-com:office:excel\"\n";
    out << " xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\"\n";
    out << " xmlns:html=\"http://www.w3.org/TR/REC-html40\">\n";

    // 1. STYLES
    out << " <Styles>\n";
    out << "  <Style ss:ID=\"Default\" ss:Name=\"Normal\">\n";
    out << "   <Alignment ss:Vertical=\"Center\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"10\" ss:Color=\"#000000\"/>\n";
    out << "  </Style>\n";

    // Title Style
    out << "  <Style ss:ID=\"TitleStyle\">\n";
    out << "   <Alignment ss:Horizontal=\"Left\" ss:Vertical=\"Center\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"14\" ss:Bold=\"1\" ss:Color=\"#1F4E78\"/>\n";
    out << "  </Style>\n";

    // Subtitle Style
    out << "  <Style ss:ID=\"SubTitleStyle\">\n";
    out << "   <Alignment ss:Horizontal=\"Left\" ss:Vertical=\"Center\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"10\" ss:Italic=\"1\" ss:Color=\"#595959\"/>\n";
    out << "  </Style>\n";

    // Header Style
    out << "  <Style ss:ID=\"HeaderStyle\">\n";
    out << "   <Alignment ss:Horizontal=\"Center\" ss:Vertical=\"Center\" ss:WrapText=\"1\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#002060\"/>\n";
    out << "    <Border ss:Position=\"Top\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#002060\"/>\n";
    out << "    <Border ss:Position=\"Left\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#002060\"/>\n";
    out << "    <Border ss:Position=\"Right\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#002060\"/>\n";
    out << "   </Borders>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"10\" ss:Bold=\"1\" ss:Color=\"#FFFFFF\"/>\n";
    out << "   <Interior ss:Color=\"#1F4E78\" ss:Pattern=\"Solid\"/>\n";
    out << "  </Style>\n";

    // Data Row Styles
    out << "  <Style ss:ID=\"DataText\">\n";
    out << "   <Alignment ss:Horizontal=\"Left\" ss:Vertical=\"Center\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataCenter\">\n";
    out << "   <Alignment ss:Horizontal=\"Center\" ss:Vertical=\"Center\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataInteger\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"#,##0\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataTimeSec\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"0.000000\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataTimeMs\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"#,##0.00\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataGflops\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"#,##0.00\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"10\" ss:Bold=\"1\" ss:Color=\"#107C41\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataSpeedup\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"0.00\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"10\" ss:Bold=\"1\" ss:Color=\"#C00000\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataEfficiency\">\n";
    out << "   <Alignment ss:Horizontal=\"Right\" ss:Vertical=\"Center\"/>\n";
    out << "   <NumberFormat ss:Format=\"0.0%\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << "  <Style ss:ID=\"DataPassed\">\n";
    out << "   <Alignment ss:Horizontal=\"Center\" ss:Vertical=\"Center\"/>\n";
    out << "   <Font ss:FontName=\"Segoe UI\" ss:Size=\"9\" ss:Bold=\"1\" ss:Color=\"#0E6251\"/>\n";
    out << "   <Interior ss:Color=\"#D4EFDF\" ss:Pattern=\"Solid\"/>\n";
    out << "   <Borders>\n";
    out << "    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\" ss:Color=\"#D9D9D9\"/>\n";
    out << "   </Borders>\n";
    out << "  </Style>\n";

    out << " </Styles>\n";

    // 2. WORKSHEET 1: TỔNG HỢP KẾT QUẢ THỰC NGHIỆM
    out << " <Worksheet ss:Name=\"Tong_Hop_Benchmark\">\n";
    out << "  <Table ss:DefaultRowHeight=\"20\">\n";
    out << "   <Column ss:Width=\"45\"/>\n";   // STT
    out << "   <Column ss:Width=\"90\"/>\n";   // Ma trận N
    out << "   <Column ss:Width=\"90\"/>\n";   // Số luồng
    out << "   <Column ss:Width=\"240\"/>\n";  // Thuật toán
    out << "   <Column ss:Width=\"80\"/>\n";   // L3 Tile
    out << "   <Column ss:Width=\"80\"/>\n";   // L2 Tile
    out << "   <Column ss:Width=\"80\"/>\n";   // L1 Tile
    out << "   <Column ss:Width=\"90\"/>\n";   // CUDA block/tile
    out << "   <Column ss:Width=\"100\"/>\n";  // Thời gian (s)
    out << "   <Column ss:Width=\"100\"/>\n";  // Thời gian (ms)
    out << "   <Column ss:Width=\"120\"/>\n";  // Tổng phép tính FLOP
    out << "   <Column ss:Width=\"100\"/>\n";  // GFLOPS
    out << "   <Column ss:Width=\"100\"/>\n";  // Tăng tốc Speedup
    out << "   <Column ss:Width=\"100\"/>\n";  // Hiệu suất Efficiency
    out << "   <Column ss:Width=\"110\"/>\n";  // Amdahl lý thuyết
    out << "   <Column ss:Width=\"110\"/>\n";  // Tỉ lệ giảm trượt Cache
    out << "   <Column ss:Width=\"100\"/>\n";  // Sai số cực đại
    out << "   <Column ss:Width=\"90\"/>\n";   // Kiểm định
    out << "   <Column ss:Width=\"140\"/>\n";  // Thời điểm chạy

    // Tiêu đề báo cáo
    out << "   <Row ss:Height=\"28\">\n";
    out << "    <Cell ss:MergeAcross=\"18\" ss:StyleID=\"TitleStyle\"><Data ss:Type=\"String\">BÁO CÁO THỰC NGHIỆM: NHÂN MA TRẬN SONG SONG CPU VÀ CUDA GPU</Data></Cell>\n";
    out << "   </Row>\n";
    out << "   <Row ss:Height=\"18\">\n";
    out << "    <Cell ss:MergeAcross=\"18\" ss:StyleID=\"SubTitleStyle\"><Data ss:Type=\"String\">CPU: "
        << escape_xml(hw.cpu_name) << " | GPU: " << escape_xml(hw.gpu_name)
        << " (Compute " << (hw.gpu_compute_major > 0
                                ? std::to_string(hw.gpu_compute_major) + "." +
                                      std::to_string(hw.gpu_compute_minor)
                                : "N/A")
        << ") | " << hw.physical_cores << " Cores / " << hw.logical_processors
        << " Threads | L2: " << hw.l2_cache_kb << "KB x " << hw.physical_cores << " | L3: " << (hw.l3_cache_kb / 1024) 
        << "MB Shared | Trình biên dịch: " << escape_xml(hw.compiler_info) << "</Data></Cell>\n";
    out << "   </Row>\n";
    out << "   <Row ss:Height=\"10\"/>\n"; // Dòng trống

    // Dòng Header
    out << "   <Row ss:Height=\"26\">\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">STT</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Kích thước N</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Số luồng (q)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thuật toán</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Khối L3 (Tile3)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Khối L2 (Tile2)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Khối L1 (Tile1)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">CUDA Block/Tile</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thời gian (s)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thời gian (ms)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Tổng FLOP (2*N³)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">GFLOPS</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Tăng tốc (Speedup)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Hiệu suất (%)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Amdahl lý thuyết</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Hệ số giảm Cache Miss</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Sai số cực đại</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Kiểm định</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thời điểm</Data></Cell>\n";
    out << "   </Row>\n";

    // Các dòng dữ liệu
    for (const auto& r : records) {
        out << "   <Row ss:Height=\"20\">\n";
        out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.id << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataInteger\"><Data ss:Type=\"Number\">" << r.N << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.threads << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << escape_xml(r.algo_name) << "</Data></Cell>\n";
        
        if (r.l3_tile > 0) out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.l3_tile << "</Data></Cell>\n";
        else out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"String\">-</Data></Cell>\n";

        if (r.l2_tile > 0) out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.l2_tile << "</Data></Cell>\n";
        else out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"String\">-</Data></Cell>\n";

        if (r.l1_tile > 0) out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.l1_tile << "</Data></Cell>\n";
        else out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"String\">-</Data></Cell>\n";

        if (r.cuda_tile > 0) out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.cuda_tile << "</Data></Cell>\n";
        else out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"String\">-</Data></Cell>\n";

        out << "    <Cell ss:StyleID=\"DataTimeSec\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(6) << r.time_sec << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataTimeMs\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.time_ms << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataInteger\"><Data ss:Type=\"Number\">" << r.flops << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataGflops\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.gflops << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataSpeedup\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.speedup << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataEfficiency\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(4) << (r.efficiency / 100.0) << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataSpeedup\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.amdahl_speedup << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataSpeedup\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(1) << r.estimated_cache_miss_reduction << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataTimeSec\"><Data ss:Type=\"Number\">" << std::scientific << std::setprecision(2) << r.max_error << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataPassed\"><Data ss:Type=\"String\">" << (r.verified ? "ĐẠT" : "KHÔNG ĐẠT") << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"String\">" << escape_xml(r.timestamp) << "</Data></Cell>\n";
        out << "   </Row>\n";
    }

    out << "  </Table>\n";
    out << " </Worksheet>\n";

    // 3. WORKSHEET 2: SO SÁNH THUẬT TOÁN (ALGORITHM COMPARISON)
    out << " <Worksheet ss:Name=\"So_Sanh_Thuat_Toan\">\n";
    out << "  <Table ss:DefaultRowHeight=\"20\">\n";
    out << "   <Column ss:Width=\"100\"/>\n";
    out << "   <Column ss:Width=\"220\"/>\n";
    out << "   <Column ss:Width=\"90\"/>\n";
    out << "   <Column ss:Width=\"110\"/>\n";
    out << "   <Column ss:Width=\"110\"/>\n";
    out << "   <Column ss:Width=\"110\"/>\n";
    out << "   <Column ss:Width=\"110\"/>\n";
    
    out << "   <Row ss:Height=\"26\">\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Kích thước N</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thuật toán</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Số luồng</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thời gian (s)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">GFLOPS</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Tăng tốc (Speedup)</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Hiệu suất (%)</Data></Cell>\n";
    out << "   </Row>\n";

    for (const auto& r : records) {
        out << "   <Row ss:Height=\"20\">\n";
        out << "    <Cell ss:StyleID=\"DataInteger\"><Data ss:Type=\"Number\">" << r.N << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << escape_xml(r.algo_name) << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataCenter\"><Data ss:Type=\"Number\">" << r.threads << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataTimeSec\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(6) << r.time_sec << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataGflops\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.gflops << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataSpeedup\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(2) << r.speedup << "</Data></Cell>\n";
        out << "    <Cell ss:StyleID=\"DataEfficiency\"><Data ss:Type=\"Number\">" << std::fixed << std::setprecision(4) << (r.efficiency / 100.0) << "</Data></Cell>\n";
        out << "   </Row>\n";
    }
    out << "  </Table>\n";
    out << " </Worksheet>\n";

    // 4. WORKSHEET 3: THÔNG TIN PHẦN CỨNG & LÝ THUYẾT CACHE
    out << " <Worksheet ss:Name=\"Thong_So_CPU_CUDA\">\n";
    out << "  <Table ss:DefaultRowHeight=\"20\">\n";
    out << "   <Column ss:Width=\"220\"/>\n";
    out << "   <Column ss:Width=\"320\"/>\n";
    out << "   <Row ss:Height=\"26\">\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Thông số kiến trúc CPU</Data></Cell>\n";
    out << "    <Cell ss:StyleID=\"HeaderStyle\"><Data ss:Type=\"String\">Giá trị chi tiết</Data></Cell>\n";
    out << "   </Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Bộ vi xử lý (CPU)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << escape_xml(hw.cpu_name) << "</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Thiết bị CUDA (GPU)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << escape_xml(hw.gpu_name) << "</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Compute capability</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << (hw.gpu_compute_major > 0 ? std::to_string(hw.gpu_compute_major) + "." + std::to_string(hw.gpu_compute_minor) : "N/A") << "</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Số nhân vật lý (Physical Cores)</Data></Cell><Cell ss:StyleID=\"DataInteger\"><Data ss:Type=\"Number\">" << hw.physical_cores << "</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Số luồng logic (Logical Processors)</Data></Cell><Cell ss:StyleID=\"DataInteger\"><Data ss:Type=\"Number\">" << hw.logical_processors << "</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">L1 Data Cache (mỗi nhân)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << hw.l1d_cache_kb << " KB (Khối Tile1 khuyến nghị: 32 - 64)</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">L2 Cache (mỗi nhân)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << hw.l2_cache_kb << " KB (Khối Tile2 khuyến nghị: 64 - 128)</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">L3 Cache (chia sẻ dùng chung)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">" << (hw.l3_cache_kb / 1024) << " MB (Khối Tile3 khuyến nghị: 256 - 512)</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Mô hình bộ nhớ</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">SMP Shared Memory (Fork-Join OpenMP)</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Tập lệnh vector hóa</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">AVX2 + FMA3 (256-bit SIMD registers)</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Định luật Amdahl (Mục 3.5)</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Speedup = 1 / ( (1 - p) + p / q )</Data></Cell></Row>\n";
    out << "   <Row><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">Tài liệu đối chiếu</Data></Cell><Cell ss:StyleID=\"DataText\"><Data ss:Type=\"String\">An - CPU song song hóa.docx (Kelefouras et al., 2016)</Data></Cell></Row>\n";
    out << "  </Table>\n";
    out << " </Worksheet>\n";

    out << "</Workbook>\n";
    out.close();
    return true;
}

bool ExcelExporter::exportToCSV(const std::string& filename,
                               const std::vector<BenchmarkRecord>& records,
                               const HardwareInfo& /*hw*/) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[Lỗi] Không thể tạo file CSV: " << filename << std::endl;
        return false;
    }

    // Header UTF-8 BOM để Excel hiển thị đúng tiếng Việt
    out << "\xEF\xBB\xBF";

    // Tiêu đề cột
    out << "STT,KichThuoc_N,SoLuong_Luong_q,ThuatToan,Khoi_L3_Tile,Khoi_L2_Tile,Khoi_L1_Tile,CUDA_Block_Tile,"
        << "ThoiGian_s,ThoiGian_ms,Tong_FLOP,HieuNang_GFLOPS,TangToc_Speedup,HieuSuat_Efficiency_Pct,"
        << "Amdahl_LyThuyet,Giam_Cache_Miss_Ratio,SaiSo_CucDai,KiemDinh,ThoiDiem\n";

    for (const auto& r : records) {
        out << r.id << ","
            << r.N << ","
            << r.threads << ","
            << "\"" << r.algo_name << "\","
            << r.l3_tile << ","
            << r.l2_tile << ","
            << r.l1_tile << ","
            << r.cuda_tile << ","
            << std::fixed << std::setprecision(6) << r.time_sec << ","
            << std::fixed << std::setprecision(2) << r.time_ms << ","
            << r.flops << ","
            << std::fixed << std::setprecision(2) << r.gflops << ","
            << std::fixed << std::setprecision(2) << r.speedup << ","
            << std::fixed << std::setprecision(2) << r.efficiency << ","
            << std::fixed << std::setprecision(2) << r.amdahl_speedup << ","
            << std::fixed << std::setprecision(1) << r.estimated_cache_miss_reduction << ","
            << std::scientific << std::setprecision(2) << r.max_error << ","
            << (r.verified ? "PASSED" : "FAILED") << ","
            << "\"" << r.timestamp << "\"\n";
    }

    out.close();
    return true;
}
