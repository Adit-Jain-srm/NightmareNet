import { NextResponse } from "next/server";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  let body: any;

  // 1. Safely attempt to parse JSON body (Catch 400 client errors)
  try {
    body = await request.json();
  } catch (parseError) {
    return NextResponse.json(
      { error: "Invalid or malformed JSON body" },
      { status: 400 }
    );
  }

  // 2. Validate request body contents
  try {
    const { name } = body || {};
    const { id } = await params;

    if (!name || typeof name !== "string" || name.trim() === "") {
      return NextResponse.json({ error: "Invalid name" }, { status: 400 });
    }

    if (name.length > 100) {
      return NextResponse.json({ error: "Name too long" }, { status: 400 });
    }

    // In a real app, this would hit the DB. For now, we mock success.
    return NextResponse.json({ success: true, id, name: name.trim() });
  } catch {
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
