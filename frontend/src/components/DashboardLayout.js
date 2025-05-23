import React from 'react';
import DashboardContent from './DashboardContent';

const DashboardLayout = ({ 
  serverStatus, 
  modelInfo, 
  loading, 
  error 
}) => {
  // Render loading state
  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-cinder">
          {/* Cinder logo as loading indicator */}
          <div className="loading-circles">
            <div className="loading-circle-1"></div>
            <div className="loading-circle-2"></div>
            <div className="loading-circle-3"></div>
          </div>
        </div>
        <div className="loading-text">Loading Model Analysis Dashboard</div>
        <div className="loading-subtext">Powered by Cinder</div>
      </div>
    );
  }

  // Render error state
  // Render error state
if (error) {
  return (
    <div className="minimalist-error-container">
      {/* Big centered Cinder logo */}
      <div className="error-logo-large">
        <div className="cinder-circles">
          <div className="cinder-circle-1"></div>
          <div className="cinder-circle-2"></div>
          <div className="cinder-circle-3"></div>
        </div>
        <span className="logo-text">Cinder</span>
      </div>
      
      {/* Pill-style error message with code format and red dot */}
      <div className="error-pill">
        <div className="status-dot"></div>
        <code className="error-code">
          Could not connect to the API server. Make sure it's running on http://localhost:8000
        </code>
      </div>
    </div>
  );
}

  return (
    <div className="dashboard">
      {/* Header with new Cinder branding */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            {/* Main Cinder Logo */}
            <div className="cinder-logo">
              <div className="cinder-circles">
                <div className="cinder-circle-1"></div>
                <div className="cinder-circle-2"></div>
                <div className="cinder-circle-3"></div>
              </div>
              <div className="logo-text">
                <h1 className="title">Cinder</h1>
              </div>
            </div>
          </div>
          
          <div className="header-center">
            {/* Model info with Bit branding */}
            {modelInfo && (
              <div className="model-info">
                <div className="bit-indicator">
                  <div className="bit-offset">
                    <div className="offset-back"></div>
                    <div class="offset-front"></div>
                  </div>
                  <span className="bit-text">Bit</span>
                </div>
                <div className="model-details">
                  <span className="model-name">{modelInfo.name}</span>
                  <span className="model-framework">{modelInfo.framework}</span>
                </div>
              </div>
            )}
          </div>

          <div className="header-buttons">
            {/* Status indicator */}
            <div className="status-indicator">
              <div className={`status-dot ${serverStatus?.status === 'online' ? 'status-online' : 'status-offline'}`}></div>
              <span className="status-text">
                {serverStatus?.status === 'online' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* Bit by Cinder branding */}
            <div className="brand-lockup">
              <div className="bit-mini">
                <div className="offset-back-mini"></div>
                <div className="offset-front-mini"></div>
              </div>
              <span className="brand-text">Bit by Cinder</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <DashboardContent 
          serverStatus={serverStatus}
          modelInfo={modelInfo}
        />
      </main>

      {/* Footer with proper branding */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-left">
            <div className="footer-brand">
              <div className="cinder-circles-small">
                <div className="cinder-circle-1-small"></div>
                <div className="cinder-circle-2-small"></div>
                <div className="cinder-circle-3-small"></div>
              </div>
              <span className="footer-brand-text">Cinder v1.0.0</span>
            </div>
          </div>
          <div className="footer-right">
            {serverStatus?.started_at && (
              <span className="footer-text">
                Server started: {new Date(serverStatus.started_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>
      </footer>

      {/* Add the CSS styles */}
      <style jsx>{`
        /* Cinder Logo Styles */
        .cinder-logo {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .cinder-circles {
          width: 40px;
          height: 40px;
          position: relative;
          flex-shrink: 0;
        }

        .cinder-circle-1 {
          width: 40px;
          height: 40px;
          background: #FF9B45;
          border-radius: 50%;
          position: absolute;
          opacity: 0.7;
        }

        .cinder-circle-2 {
          width: 28px;
          height: 28px;
          background: #D5451B;
          border-radius: 50%;
          position: absolute;
          top: 6px;
          left: 6px;
          opacity: 0.8;
        }

        .cinder-circle-3 {
          width: 12px;
          height: 12px;
          background: #521C0D;
          border-radius: 50%;
          position: absolute;
          top: 14px;
          left: 14px;
        }

        .logo-text .title {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
          color: #D5451B;
          letter-spacing: -0.5px;
          line-height: 1;
        }

        .logo-text .subtitle {
          margin: 4px 0 0 0;
          font-size: 14px;
          color: #6b7280;
          font-weight: 400;
        }

        /* Bit Indicator Styles */
        .bit-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(213, 69, 27, 0.1);
          padding: 8px 12px;
          border-radius: 8px;
          border: 1px solid rgba(213, 69, 27, 0.2);
        }

        .bit-offset {
          width: 20px;
          height: 20px;
          position: relative;
        }

        .offset-back {
          width: 18px;
          height: 18px;
          background: #FF9B45;
          border-radius: 4px;
          position: absolute;
          bottom: 0;
          right: 0;
          opacity: 0.6;
        }

        .offset-front {
          width: 18px;
          height: 18px;
          background: #D5451B;
          border-radius: 4px;
          position: absolute;
          top: 0;
          left: 0;
        }

        .bit-text {
          font-size: 14px;
          font-weight: 600;
          color: #D5451B;
        }

        .model-details {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .model-name {
          font-size: 14px;
          font-weight: 600;
          color: #111827;
        }

        .model-framework {
          font-size: 12px;
          color: #6b7280;
          text-transform: uppercase;
          font-weight: 500;
        }

        /* Header layout improvements */
        .header-content {
          display: flex;
          align-items: center;
          justify-content: space-between;
          max-width: 1280px;
          margin: 0 auto;
          padding: 16px 24px;
          gap: 24px;
        }

        .header-center {
          flex: 1;
          display: flex;
          justify-content: center;
        }

        .model-info {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        /* Brand lockup */
        .brand-lockup {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #f8f9fa;
          padding: 6px 12px;
          border-radius: 20px;
          border: 1px solid #e5e7eb;
        }

        .bit-mini {
          width: 16px;
          height: 16px;
          position: relative;
        }

        .offset-back-mini {
          width: 14px;
          height: 14px;
          background: #FF9B45;
          border-radius: 3px;
          position: absolute;
          bottom: 0;
          right: 0;
          opacity: 0.6;
        }

        .offset-front-mini {
          width: 14px;
          height: 14px;
          background: #D5451B;
          border-radius: 3px;
          position: absolute;
          top: 0;
          left: 0;
        }

        .brand-text {
          font-size: 13px;
          font-weight: 500;
          color: #D5451B;
        }

        /* Status indicator improvements */
        .status-indicator {
          display: flex;
          align-items: center;
          background: #f8f9fa;
          padding: 8px 12px;
          border-radius: 20px;
          border: 1px solid #e5e7eb;
          gap: 8px;
        }

        .status-text {
          font-size: 13px;
          font-weight: 500;
          color: #374151;
        }

        /* Loading state with Cinder logo */
        .loading-cinder {
          width: 60px;
          height: 60px;
          position: relative;
          animation: pulse 2s ease-in-out infinite;
        }

        .loading-circles {
          width: 60px;
          height: 60px;
          position: relative;
        }

        .loading-circle-1 {
          width: 60px;
          height: 60px;
          background: #FF9B45;
          border-radius: 50%;
          position: absolute;
          opacity: 0.7;
        }

        .loading-circle-2 {
          width: 42px;
          height: 42px;
          background: #D5451B;
          border-radius: 50%;
          position: absolute;
          top: 9px;
          left: 9px;
          opacity: 0.8;
        }

        .loading-circle-3 {
          width: 18px;
          height: 18px;
          background: #521C0D;
          border-radius: 50%;
          position: absolute;
          top: 21px;
          left: 21px;
        }

        .loading-subtext {
          margin-top: 8px;
          font-size: 14px;
          color: #6b7280;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.05); }
        }

        /* Error state */
        .error-header {
          margin-bottom: 24px;
          display: flex;
          justify-content: center;
        }

        .error-logo {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .error-brand {
          font-size: 20px;
          font-weight: 600;
          color: #D5451B;
        }

        /* Footer improvements */
        .footer-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
          max-width: 1280px;
          margin: 0 auto;
          padding: 16px 24px;
        }

        .footer-brand {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .cinder-circles-small {
          width: 16px;
          height: 16px;
          position: relative;
        }

        .cinder-circle-1-small {
          width: 16px;
          height: 16px;
          background: #FF9B45;
          border-radius: 50%;
          position: absolute;
          opacity: 0.7;
        }

        .cinder-circle-2-small {
          width: 11px;
          height: 11px;
          background: #D5451B;
          border-radius: 50%;
          position: absolute;
          top: 2.5px;
          left: 2.5px;
          opacity: 0.8;
        }

        .cinder-circle-3-small {
          width: 5px;
          height: 5px;
          background: #521C0D;
          border-radius: 50%;
          position: absolute;
          top: 5.5px;
          left: 5.5px;
        }

        .footer-brand-text {
          font-size: 14px;
          font-weight: 500;
          color: #D5451B;
        }

        .footer-text {
          font-size: 13px;
          color: #6b7280;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .header-content {
            flex-direction: column;
            gap: 16px;
          }

          .header-center {
            order: -1;
            width: 100%;
          }

          .model-info {
            justify-content: center;
          }

          .footer-content {
            flex-direction: column;
            gap: 12px;
            text-align: center;
          }
        }

        @media (max-width: 480px) {
          .cinder-logo {
            gap: 12px;
          }

          .cinder-circles {
            width: 32px;
            height: 32px;
          }

          .cinder-circle-1 {
            width: 32px;
            height: 32px;
          }

          .cinder-circle-2 {
            width: 22px;
            height: 22px;
            top: 5px;
            left: 5px;
          }

          .cinder-circle-3 {
            width: 10px;
            height: 10px;
            top: 11px;
            left: 11px;
          }

          .logo-text .title {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default DashboardLayout;