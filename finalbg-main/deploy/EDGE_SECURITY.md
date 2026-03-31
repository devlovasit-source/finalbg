# Edge Security Setup

This backend is intended to run behind the `gateway` service (Nginx), plus optional Cloudflare/AWS API Gateway.

## 1) TLS Termination (Nginx)
- Place cert files at:
  - `deploy/nginx/certs/fullchain.pem`
  - `deploy/nginx/certs/privkey.pem`
- Start stack:
  - `docker compose up -d --build`

## 2) Gateway Protections Enabled
- HTTPS redirect (80 -> 443)
- `client_max_body_size 5m`
- Per-IP rate limiting: `60 req/min`
- Per-IP connection limiting
- Secure headers (HSTS, X-Frame-Options, etc.)

## 3) Cloudflare / API Gateway (Recommended)
- Enable WAF managed rules
- Enable DDoS protection
- Set max upload size to 5MB
- Add geo/IP reputation blocks where needed
- Add bot challenge for suspicious traffic

## 4) Origin Lockdown
- Restrict inbound security group/firewall so only gateway/edge can reach app ports.
- Do not expose Uvicorn directly to the internet.
