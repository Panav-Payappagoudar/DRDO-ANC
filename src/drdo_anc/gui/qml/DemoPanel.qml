import QtQuick
import QtQuick.Layouts

Item {
    id: root
    property color white: "#F0F4F8"
    property color cyan: "#00E5FF"
    property color dim: "#888888"
    property color panel: "#11151C"

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        Text {
            text: "DEMO STATUS"
            color: dim
            font.pixelSize: 9
            font.bold: true
        }

        GridLayout {
            columns: 2
            columnSpacing: 12
            rowSpacing: 2
            Layout.fillWidth: true

            Text { text: "Model"; color: dim; font.pixelSize: 10 }
            Text { text: guiBridge.modelName; color: white; font.pixelSize: 10 }

            Text { text: "Sample rate"; color: dim; font.pixelSize: 10 }
            Text { text: guiBridge.sampleRate + " Hz"; color: white; font.pixelSize: 10 }

            Text { text: "Latency"; color: dim; font.pixelSize: 10 }
            Text { text: guiBridge.processingTimeMs.toFixed(2) + " ms"; color: white; font.pixelSize: 10 }

            Text { text: "RTF"; color: dim; font.pixelSize: 10 }
            Text { text: guiBridge.realtimeFactor.toFixed(2) + "x"; color: white; font.pixelSize: 10 }

            Text { text: "Audio"; color: dim; font.pixelSize: 10 }
            Text { text: guiBridge.audioStatus; color: cyan; font.pixelSize: 10 }

            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoNoisyFile.length > 0
                text: "Noisy input"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoNoisyFile.length > 0
                text: guiBridge.demoNoisyFile
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoCleanFile.length > 0
                text: "Clean ref"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoCleanFile.length > 0
                text: guiBridge.demoCleanFile
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoEnhancedRefFile.length > 0
                text: "Enh. ref"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.operationMode === "demo" && guiBridge.demoEnhancedRefFile.length > 0
                text: guiBridge.demoEnhancedRefFile
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.operationMode === "demo"
                text: "B playback"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.operationMode === "demo"
                text: guiBridge.demoEnhancedPlayback
                color: cyan
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.showDemoMetrics
                text: "Noisy SNR"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.showDemoMetrics
                text: guiBridge.demoNoisySnr.toFixed(1) + " dB"
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.showDemoMetrics
                text: "Ref enh SNR"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.showDemoMetrics
                text: guiBridge.demoEnhancedRefSnr.toFixed(1) + " dB"
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.showBenchmarkSummary
                text: "Development cases"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.showBenchmarkSummary
                text: guiBridge.developmentCases
                color: white
                font.pixelSize: 10
            }

            Text {
                visible: guiBridge.showBenchmarkSummary
                text: "Evaluations"
                color: dim
                font.pixelSize: 10
            }
            Text {
                visible: guiBridge.showBenchmarkSummary
                text: guiBridge.evaluations
                color: white
                font.pixelSize: 10
            }
        }
    }
}
