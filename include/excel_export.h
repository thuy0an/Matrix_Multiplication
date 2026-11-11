#ifndef EXCEL_EXPORT_H
#define EXCEL_EXPORT_H

#include "common.h"
#include <string>
#include <vector>

class ExcelExporter {
public:
    // Xuất file Excel định dạng XML Spreadsheet 2003 (.xls) mở trực tiếp bằng Microsoft Excel
    // Có đầy đủ định dạng bảng, màu sắc header, định dạng số, công thức và nhiều sheet
    static bool exportToExcel(const std::string& filename, 
                             const std::vector<BenchmarkRecord>& records,
                             const HardwareInfo& hw);

    // Xuất file CSV tiêu chuẩn (.csv) để tiện import vào Python, R, SPSS, MATLAB
    static bool exportToCSV(const std::string& filename,
                           const std::vector<BenchmarkRecord>& records,
                           const HardwareInfo& hw);
};

#endif // EXCEL_EXPORT_H
