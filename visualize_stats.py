#!/usr/bin/env python3
"""Visualize graph statistics using PySide6."""
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QTabWidget, QTableWidget, 
                               QTableWidgetItem, QTextEdit, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtCharts import (QChart, QChartView, QBarSeries, QBarSet, 
                              QBarCategoryAxis, QValueAxis, QPieSeries, QPieSlice)
from PySide6.QtGui import QColor


class GraphStatsVisualizer(QMainWindow):
    """Main window for visualizing graph statistics."""
    
    def __init__(self, nodes_path, edges_path, types_path=None):
        super().__init__()
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.types_path = types_path
        
        self.setWindowTitle("TriVector Code Intelligence - Graph Statistics")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load data
        self.nodes = []
        self.edges = []
        self.types = []
        
        self.load_data()
        self.init_ui()
    
    def load_data(self):
        """Load data from JSONL files."""
        # Load nodes
        with open(self.nodes_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.nodes.append(json.loads(line))
        
        # Load edges
        with open(self.edges_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.edges.append(json.loads(line))
        
        # Load types if available
        if self.types_path and Path(self.types_path).exists():
            with open(self.types_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.types.append(json.loads(line))
    
    def init_ui(self):
        """Initialize the UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Graph Statistics Visualization")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Tabs
        tabs = QTabWidget()
        
        # Node statistics tab
        tabs.addTab(self.create_nodes_tab(), "Nodes")
        
        # Edge statistics tab
        tabs.addTab(self.create_edges_tab(), "Edges")
        
        # Type statistics tab
        if self.types:
            tabs.addTab(self.create_types_tab(), "Types")
        
        # Summary tab
        tabs.addTab(self.create_summary_tab(), "Summary")
        
        layout.addWidget(tabs)
    
    def create_nodes_tab(self):
        """Create nodes statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Node kind distribution
        kind_counts = Counter(node.get('kind', 'unknown') for node in self.nodes)
        
        # Chart
        chart = QChart()
        chart.setTitle("Node Kind Distribution")
        
        series = QBarSeries()
        bar_set = QBarSet("Count")
        
        categories = []
        for kind, count in kind_counts.most_common():
            bar_set.append(count)
            categories.append(kind)
        
        series.append(bar_set)
        chart.addSeries(series)
        
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(kind_counts.values()) * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setMinimumHeight(300)
        layout.addWidget(chart_view)
        
        # Table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Kind", "Count"])
        table.setRowCount(len(kind_counts))
        
        for i, (kind, count) in enumerate(kind_counts.most_common()):
            table.setItem(i, 0, QTableWidgetItem(kind))
            table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        return widget
    
    def create_edges_tab(self):
        """Create edges statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Relationship type distribution
        rel_counts = Counter(edge.get('rel', 'unknown') for edge in self.edges)
        
        # Chart
        chart = QChart()
        chart.setTitle("Edge Relationship Distribution")
        
        series = QBarSeries()
        bar_set = QBarSet("Count")
        
        categories = []
        for rel, count in rel_counts.most_common():
            bar_set.append(count)
            categories.append(rel)
        
        series.append(bar_set)
        chart.addSeries(series)
        
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(rel_counts.values()) * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setMinimumHeight(300)
        layout.addWidget(chart_view)
        
        # Weight statistics
        weights = [edge.get('weight', 1.0) for edge in self.edges]
        if weights:
            weight_stats = QTextEdit()
            weight_stats.setReadOnly(True)
            weight_stats.append(f"Total Edges: {len(self.edges)}")
            weight_stats.append(f"Average Weight: {np.mean(weights):.4f}")
            weight_stats.append(f"Min Weight: {np.min(weights):.4f}")
            weight_stats.append(f"Max Weight: {np.max(weights):.4f}")
            weight_stats.append(f"Std Weight: {np.std(weights):.4f}")
            layout.addWidget(weight_stats)
        
        # Table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Relationship", "Count"])
        table.setRowCount(len(rel_counts))
        
        for i, (rel, count) in enumerate(rel_counts.most_common()):
            table.setItem(i, 0, QTableWidgetItem(rel))
            table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        return widget
    
    def create_types_tab(self):
        """Create type statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Type token distribution
        type_counts = Counter(type_entry.get('type_token', 'unknown') for type_entry in self.types)
        
        # Chart (top 20)
        chart = QChart()
        chart.setTitle("Top 20 Type Tokens")
        
        series = QBarSeries()
        bar_set = QBarSet("Count")
        
        categories = []
        for type_token, count in type_counts.most_common(20):
            bar_set.append(count)
            categories.append(type_token[:20])  # Truncate long names
        
        series.append(bar_set)
        chart.addSeries(series)
        
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        if type_counts:
            axis_y.setRange(0, max(type_counts.values()) * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setMinimumHeight(300)
        layout.addWidget(chart_view)
        
        # Table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Type Token", "Count"])
        table.setRowCount(len(type_counts))
        
        for i, (type_token, count) in enumerate(type_counts.most_common()):
            table.setItem(i, 0, QTableWidgetItem(type_token))
            table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        return widget
    
    def create_summary_tab(self):
        """Create summary statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setStyleSheet("font-family: monospace; font-size: 12px;")
        
        # Basic stats
        summary.append("=" * 60)
        summary.append("GRAPH STATISTICS SUMMARY")
        summary.append("=" * 60)
        summary.append("")
        
        summary.append(f"Total Nodes: {len(self.nodes)}")
        summary.append(f"Total Edges: {len(self.edges)}")
        if self.types:
            summary.append(f"Total Type Tokens: {len(set(t.get('type_token') for t in self.types))}")
        summary.append("")
        
        # Node stats
        summary.append("NODE STATISTICS:")
        summary.append("-" * 60)
        kind_counts = Counter(node.get('kind', 'unknown') for node in self.nodes)
        for kind, count in kind_counts.most_common():
            percentage = (count / len(self.nodes)) * 100
            summary.append(f"  {kind:20s}: {count:6d} ({percentage:5.2f}%)")
        summary.append("")
        
        # Edge stats
        summary.append("EDGE STATISTICS:")
        summary.append("-" * 60)
        rel_counts = Counter(edge.get('rel', 'unknown') for edge in self.edges)
        for rel, count in rel_counts.most_common():
            percentage = (count / len(self.edges)) * 100
            summary.append(f"  {rel:20s}: {count:6d} ({percentage:5.2f}%)")
        summary.append("")
        
        # Weight stats
        weights = [edge.get('weight', 1.0) for edge in self.edges]
        if weights:
            summary.append("WEIGHT STATISTICS:")
            summary.append("-" * 60)
            summary.append(f"  Mean: {np.mean(weights):.4f}")
            summary.append(f"  Median: {np.median(weights):.4f}")
            summary.append(f"  Std:   {np.std(weights):.4f}")
            summary.append(f"  Min:   {np.min(weights):.4f}")
            summary.append(f"  Max:   {np.max(weights):.4f}")
            summary.append("")
        
        # Type stats
        if self.types:
            summary.append("TYPE STATISTICS:")
            summary.append("-" * 60)
            type_counts = Counter(type_entry.get('type_token', 'unknown') for type_entry in self.types)
            summary.append(f"  Unique Types: {len(type_counts)}")
            summary.append(f"  Total Type Entries: {len(self.types)}")
            summary.append("")
            summary.append("  Top 10 Type Tokens:")
            for i, (type_token, count) in enumerate(type_counts.most_common(10), 1):
                summary.append(f"    {i:2d}. {type_token:30s}: {count:6d}")
        
        layout.addWidget(summary)
        
        return widget


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize graph statistics')
    parser.add_argument('--nodes', required=True, help='Path to nodes.jsonl')
    parser.add_argument('--edges', required=True, help='Path to edges.jsonl')
    parser.add_argument('--types', default=None, help='Path to types.jsonl (optional)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    window = GraphStatsVisualizer(args.nodes, args.edges, args.types)
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

