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
 fix/issue-462-reconcile-rename
    const { name } = body || {};

    const { id } = await params;
    const body = await request.json();
    const { name } = body;
    main

    if (!name || typeof name !== "string" || name.trim() === "") {
      return NextResponse.json({ error: "Invalid name" }, { status: 400 });
    }

    if (name.length > 100) {
      return NextResponse.json({ error: "Name too long" }, { status: 400 });
    }

 fix/issue-462-reconcile-rename
    // In a real app, this would hit the DB. For now, we mock success.
    return NextResponse.json({
      success: true,
      id: params.id,
      name: name.trim(),
    });

    return NextResponse.json({ success: true, id, name: name.trim() });
 main
  } catch (error) {
    // 3. Any unexpected server/database errors fall back to 500
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}